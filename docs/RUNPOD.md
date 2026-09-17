# RunPod

GPU lane for the [TabPFN-3.5 transfer backlog](../TODO.md). Ported from the pattern already
in use in `black_swan` and `ask_max` — same account, same API quirks — and re-aimed at this
project's two workloads.

```
python scripts/cloud/runpod_launch.py check              # credential + what is already billing
python scripts/cloud/runpod_launch.py gpus --min-vram 48 # prices and stock, live
python scripts/cloud/runpod_launch.py plan --lane train  # the exact create body and $/hr, no call
```

Everything except `create` is read-only. `create` refuses without `--yes-i-will-pay`.

## Why a GPU is needed at all

Phase 0's harness and everything in Phase 1 run on a laptop. What does not:

| Work | Why it needs a GPU |
|---|---|
| **BM-02 full sweep** | The 1M-row end of the inference curve. Memory-bound — see below. |
| **BM-06** proxy validation | Sign-agreement runs against known TabICLv2 ablations. Training. |
| **Phase 2–4 ablations** | Every one is a proxy-scale pre-training run. |
| Full curriculum | 3 stages × 500K/40K/10K steps. Four GPUs, and not something to start before the proxy has picked the winners. |

## Two lanes, two cards

`--lane` picks the preset; `--gpu` overrides it and disables substitution.

### `sweep` — BM-02, memory-bound, hours

The KV cache is **48 KiB per training row per estimator** ([RESULTS.md](../benchmarks/RESULTS.md)),
so the top of the curve is bounded by VRAM long before it is bounded by FLOPs. In fp32 at
8 estimators that is ~37 GiB at 100K rows; fp16 halves it, but 1M rows does not fit on
anything at `kv_cache="kv"` and will need `"repr"` mode or a truncated sweep.

Default card is **RTX PRO 6000 (96 GB)** — deliberately the card the TabPFN-3.5 report used
for its own Figure 7, so our curve lands on the same hardware as the one we are trying to
beat instead of needing a cross-card correction.

### `train` — BM-06 and the ablations, cost-bound, many hours × many runs

The proxy model is small: `embed_dim 128`, 12 ICL blocks, `d_model 512`. It does not need
96 GB. What it does want is **FlashAttention-3**, which is sm_90+ — stages 2 and 3 of the v2
recipe pass `--use_flash_attn3 True` and
[fall back silently](../src/tabicl/_model/attention.py) without it. That is why an H100 leads
this lane despite an A100 being cheaper: a stage-2 run on an A100 is a *different run* from
the same script on an H100, with nothing in the log to say so. `pod_doctor.py` reports this.

**Never mix cards inside one measured comparison.** A treatment on an H100 against a control
on an A100 is a hardware difference wearing an ablation's clothes. `pod_doctor.json` records
`gpu`, `sm` and `vram_gb` so a result stays attributable — which only helps if you read it.

## Prices

Read from the API **2026-09-17**. They drift; re-read with `gpus` rather than trusting this.

| GPU | VRAM | sm | secure OD | community OD | FA3? |
|---|---|---|---|---|---|
| H200 SXM | 141 GB | 90 | $4.59 | $3.59 | yes |
| H100 NVL | 94 GB | 90 | $3.19 | $2.59 | yes |
| **RTX PRO 6000** | **96 GB** | **120** | **$2.09** | **$1.69** | partial |
| RTX PRO 6000 MaxQ | 96 GB | 120 | **$0.50** | $1.64 | partial |
| **H100 PCIe** | **80 GB** | **90** | **$2.89** | **$1.99** | yes |
| A100 PCIe | 80 GB | 80 | $1.59 | $1.19 | no |
| L40S | 48 GB | 89 | $1.09 | $0.79 | no |
| RTX A6000 | 48 GB | 86 | $0.53 | $0.33 | no |

Two things in that table are worth a second look:

- **RTX PRO 6000 MaxQ at $0.50 secure** is the cheapest 96 GB on the board, cheaper than its
  own community listing. MaxQ is the ~300 W variant, so it is slower — but the `sweep` lane
  is memory-bound, not compute-bound, which is exactly the case where trading clocks for
  VRAM per dollar is free. Worth trying before paying $1.69. Capped at 2 GPUs.
- **Blackwell FA3 support is "partial", not "yes".** FlashAttention-3 targets Hopper (sm_90);
  sm_120 support exists in recent builds but is not guaranteed. `pod_doctor.py` reports what
  actually imports rather than inferring it from the compute capability alone.

At the time of writing, stock for RTX PRO 6000 and H100 PCIe was **zero** in RunPod's listed
datacenters. `gpuTypePriority: availability` means RunPod walks the lane's alternates rather
than failing, and community hosts outside RunPod's own datacenters do not appear in the
`dataCenters` map at all — `(no listed DC)` is not the same as unavailable.

## End to end

```bash
python scripts/cloud/runpod_launch.py check
python scripts/cloud/runpod_launch.py plan --lane train     # review this
python scripts/cloud/runpod_launch.py create --lane train --yes-i-will-pay
python scripts/cloud/runpod_launch.py payload               # builds ../tabicl_payload.tgz
python scripts/cloud/runpod_launch.py ssh POD_ID            # prints the scp/ssh lines
```

Then, on the pod:

```bash
cd /workspace && tar xzf tabicl_payload.tgz
bash tabicl/scripts/cloud/runpod_bootstrap.sh     # INSTALL_FA3=1 to build flash-attn
```

The bootstrap installs the checkout **editable**, runs `pod_doctor.py`, warms the released
checkpoint cache, and stops. It deliberately does not start a long run: a bootstrap failure
should cost one minute of GPU time, not be discovered an hour into a sweep.

Pull results before terminating — the ledger is the deliverable, everything else is
reproducible:

```bash
scp -P PORT -r root@IP:/workspace/tabicl/benchmarks/_results ./benchmarks/_results.pod
scp -P PORT root@IP:/workspace/tabicl/pod_doctor.json ./pod_doctor.json
```

## Cost control

`stop` keeps the volume **and keeps billing it** (~$0.012/hr for 100 GB). `terminate`
destroys it and needs `--yes-destroy-the-volume`.

`check` prints balance, spend limit and every live pod in one output, on purpose: a pod
someone forgot about is the most expensive thing this account can do, and it is invisible
from a terminal that only ever runs `create`.

## The API, as actually encountered

Inherited from `black_swan/docs/RUNPOD.md` and re-verified here on 2026-09-17:

- **Two API surfaces, both needed.** REST (`rest.runpod.io/v1`) for pods and templates;
  GraphQL (`api.runpod.io/graphql`) for prices, stock and `myself`, none of which REST
  exposes.
- **GraphQL sits behind Cloudflare, which returns 403 error 1010 when there is no
  User-Agent header** — indistinguishable from a rejected key. `_request` always sets one.
- **`lowestPrice` returns nothing when stock is zero**, so a plan for an out-of-stock card
  printed `$None/hr` — exactly when you most want to know what it would cost if it came
  back. `plan` now falls back to the catalogue price and says that is what it did.
- **The credential is `~/runpod.token`** (`RUNPOD_TOKEN_FILE` / `RUNPOD_API_KEY` override),
  read at the point of use, never echoed, never written into this repo.
- **`runpodctl` does not read `RUNPOD_API_KEY`.** With no key configured it prints an empty
  pod list and exits 0 — it looks like "you have no pods", not "I am not authenticated".
