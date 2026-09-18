# Results ledger

Single source of truth for every measurement taken against the
[TabPFN-3.5 transfer backlog](../TODO.md). See
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for the protocol.

## Rules

1. **Append only.** Rows are never edited or deleted. A superseded result is superseded by a new
   row that references the old one, not by rewriting it.
2. **Provenance is mandatory.** Commit SHA, checkpoint identity, seed, hardware. A number without
   provenance is not a result.
3. **Cost is recorded alongside accuracy.** Wall-clock and peak VRAM. The target is a Pareto
   frontier, not a single number.
4. **Negative results are recorded with the same care as positive ones**, with a one-line reading
   of why. Knowing which of TabPFN's changes do *not* transfer is half the value of this work.
5. **≥3 seeds** for anything that gates a phase decision.

---

## Reference points (external, not ours)

From the TabPFN-3.5 technical report, 14 Sep 2026. Recorded so our numbers have something to be
compared against. **These were not measured by us** and the BeyondArena values are read off
Figures 3–4 (labels rounded to the nearest 10, wide CIs on small slices).

| Benchmark | Metric | TabICLv2 | TabPFN-3 | TabPFN-3.5 | Source |
|---|---|---|---|---|---|
| BeyondArena (all) | Elo | ~1200 | ~1280 | ~1420 | Fig. 3 |
| BeyondArena (high-card) | Elo | ~1180 | ~1290 | ~1340 | Fig. 3 |
| BeyondArena (text) | Elo | ~1230 | ~1180 | ~1580 | Fig. 3 |
| STRABLE (TF-IDF) | Elo | 1383 | — | 1714 | Fig. 5a |
| MulTaBench | Elo | 1637 | — | 1856 | Fig. 5b |
| ScoringBench | CRPS mean rank | 12.05 *(finetuned)* | 7.64 | 2.85 | Fig. 22 |
| fev-bench | SQL skill | not entered | 43.1 | 44.2 | Fig. 21 |

---

## BM-05 — Baseline (current released checkpoints)

**Status: measured 2026-09-17.** RunPod pod `2ne93b4ii3qfkq`, **NVIDIA RTX PRO 6000 Blackwell
Server Edition, 95 GB, sm_120, 128 vCPU**, torch 2.8.0+cu128, checkpoint
`tabicl-classifier-v2-20260212`, commit `23e8e88`, `n_estimators=8`.

The card is deliberately the one the TabPFN-3.5 report used for its own Figure 7, so these
numbers sit on the same hardware as the ones we are trying to beat. FlashAttention-3 was NOT
installed (hardware-capable, sm_120); irrelevant for inference, relevant for any stage-2/3
training run on this pod.

Pod cost for everything in this section: **$1.32**.

### Speed and memory (BM-02)

| Rows | fit+predict (s) | cached 1-row (s) | cached 100-row (s) | cache (GiB) | KiB/row/est |
|---|---|---|---|---|---|
| 1,000 | 0.70 | 0.0419 | 0.0278 | 0.34 | 44.0 |
| 2,000 | 0.81 | 0.0412 | 0.0265 | 0.52 | 34.0 |
| 4,000 | 1.09 | 0.0413 | 0.0263 | 0.88 | 29.0 |
| 8,000 | 1.63 | 0.0419 | 0.0267 | 1.62 | 26.5 |
| 16,000 | 2.60 | 0.0412 | 0.0266 | 3.08 | 25.2 |
| 32,000 | 4.82 | 0.0413 | 0.0264 | 6.01 | 24.6 |
| 64,000 | 9.92 | 0.0420 | 0.0298 | 11.87 | 24.3 |
| 128,000 | 23.68 | 0.0479 | 0.0378 | 23.59 | 24.2 |
| 256,000 | 98.47 | 0.0627 | 0.0538 | 47.03 | 24.1 |

Zero errors. 100 columns, 1,024 test rows for the uncached path, median of 3 repeats
(1 repeat at 256k).

#### The cache law, confirmed at scale

Cache per row per estimator converges to **24.1 KiB**, and the earlier CPU smoke test measured
**48 KiB in fp32**. That is exactly 2x: the GPU path runs under AMP in fp16. The prediction
made from two points on a laptop holds across three orders of magnitude on real hardware, and
the residual at small row counts is the fixed term amortizing, not a different law.

`12 ICL blocks x 2 (K+V) x 512 dim x 2 bytes = 24 KiB`.

**Ceiling on this card:** 47 GiB at 256k rows. 512k would need ~94 GiB against 95 GB of VRAM
and did not complete within the session. So `kv_cache="kv"` tops out somewhere around
**400-500k training rows on a 96 GB card**, at 8 estimators, today. TP-05 (grouped-query
attention) would cut that by ~8x; TP-06 (doubled width) would halve it again without TP-05.

#### Finding: single-row cached prediction is *slower* than 100-row

| Rows | 1 test row | 100 test rows | ratio |
|---|---|---|---|
| 1,000 | 0.0419 | 0.0278 | **1.51x slower** |
| 32,000 | 0.0413 | 0.0264 | **1.56x slower** |
| 256,000 | 0.0627 | 0.0538 | 1.17x slower |

Predicting **one** row costs ~1.5x predicting a hundred. Per-call overhead dominates the
online path, so the marginal cost of the first 99 rows is negative. This is not a rounding
artefact — it holds at every row count, across 3 repeats.

It matters because the single-test-row case is exactly what the report highlights for cached
inference ("a single test example is scored against the training context"), and it is the
shape of real online serving. Worth a profile: it is a fixed cost, so it is likely Python-side
setup, preprocessing, or an ensembling loop rather than attention.

**New backlog item.** Not from the report — found here.

#### Cached prediction does not scale with training rows

0.041 s at 1k rows, 0.042 s at 64k, 0.048 s at 128k, 0.063 s at 256k. Essentially flat over a
256x range. The KV cache delivers what it promises on latency; its cost is memory, and that
is where TP-05 applies.

#### Quadratic onset above ~64k rows

Uncached fit+predict: 9.92 s at 64k, 23.68 s at 128k (2.4x for 2x rows), 98.47 s at 256k
(**4.2x for 2x rows**). Consistent with the report's note that these scale quadratically past
128k, and it lands slightly earlier here.

### Real-data accuracy (BM-03)

10 frozen OpenML datasets, 3 folds, capped at 3,000 rows. 120 rows, **zero errors**, 48 s
total.

| Dataset | accuracy | balanced acc | ROC-AUC | log-loss |
|---|---|---|---|---|
| banknote-authentication | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| blood-transfusion | 0.7954 | 0.6298 | 0.7580 | 0.4684 |
| credit-g | 0.7590 | 0.6726 | 0.8054 | 0.4783 |
| diabetes | 0.7669 | 0.7248 | 0.8437 | 0.4654 |
| kc1 | 0.8672 | 0.6307 | 0.8537 | 0.3156 |
| phoneme | 0.9000 | 0.8713 | 0.9586 | 0.2354 |
| qsar-biodeg | 0.8711 | 0.8483 | 0.9376 | 0.3014 |
| steel-plates-fault | 1.0000 | 1.0000 | 1.0000 | 0.0001 |
| vehicle | 0.8853 | 0.8868 | 0.9791 | 0.2337 |
| wdbc | 0.9719 | 0.9700 | 0.9962 | 0.0693 |

**Two datasets are saturated and should be treated as dead weight:**
`banknote-authentication` (1.0000/1.0000/1.0000) and `steel-plates-fault`
(1.0000, log-loss 0.0001). A dataset every method solves perfectly contributes nothing to a
rank or an Elo and only dilutes the mean. They are kept — the suite is frozen, and silently
dropping datasets after seeing results is how benchmarks become flattering — but they must be
excluded from, or reported separately in, any ablation readout. Recorded as **BM-07** in
[TODO.md](../TODO.md).

---

## BM-06 — Proxy-scale training: throughput and cost

**Status: throughput measured, calibration NOT run.** This is the section that changes the plan.

Clean runs, nothing else on the GPU, 40 steps each, `NJOBS=16`:

| micro_batch_size | batch_size_per_gp | s/it |
|---|---|---|
| 4 (recipe default) | 4 | 3.11 |
| 8 | 8 | 3.04 |
| 16 | 16 | 3.00 |

**Micro-batching is not the bottleneck.** Tripling it buys 3.5%. ~3.0 s/it is the floor for
this configuration on a 96 GB Blackwell card, and `prior_time` was 0 after the first step, so
the CPU prior is keeping up — the step is GPU-bound at `batch_size 64 x 1024 samples`.

### The cost problem

| | s/it | hours | $ at $2.09/hr |
|---|---|---|---|
| One 25,000-step proxy run | 3.0 | **20.8** | **$43.50** |
| BM-06 calibration (control + 3 knobs) | | 83 | **$174** |
| Phase 2 at 3 seeds (3 ablations + control, x3) | | ~250 | **~$520** |

Against an account balance of **$98.33**, the ablation programme as specified in
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) is **not affordable** — Phase 2 alone is
roughly five times the balance, and that is before Phases 3 and 4.

This is precisely what BM-06 existed to discover, and it is better to have discovered it for
$1.32 than 80 hours in. It does not invalidate the plan; it re-prices it. Options, in the order
worth trying:

1. **Find the cheapest step count that still passes sign agreement.** The plan fixes 25,000
   steps as "~5% of Stage 1", which was a guess, not a measurement. Run the calibration at
   5,000 steps first ($8.70/run, 4 runs = $35). If the signs agree there, the whole programme
   drops by 5x. Only raise steps if they do not.
2. **Move training to a cheaper card.** This lane does not need 96 GB — the proxy model is
   `embed_dim 128`, `d_model 512`. An RTX A6000 is $0.33/hr against $2.09. Even at 3x the
   wall-clock that is a 2x saving, and it costs FlashAttention-3 (sm_86), which only matters
   for stage 2/3 recipes. **Never mix cards within one comparison** — re-run the control on
   whichever card the treatment used.
3. **Cut seeds from 3 to 2 for screening**, keeping 3 only for whatever survives to a
   full-scale run.

### Two bugs found by running it

- **`NJOBS = cpu_count - 2` is catastrophic on a large pod.** 128 vCPUs gave 126 prior-worker
  processes, each opening a 64-thread OpenBLAS pool; the run died with
  `pthread_create failed ... Resource temporarily unavailable` before step 1. The visible
  error is a DataLoader worker "killed by signal: Interrupt", which points nowhere near the
  cause. Fixed in `scripts/train_proxy.sh`: cap at 32 workers and pin
  `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
- **`micro_batch_size` cannot exceed `batch_size_per_gp` when `seq_len_per_gp=True`**, because
  every dataset in a micro-batch must share a training size
  (`ValueError: All datasets in the micro batch must have the same training size`). Raising one
  without the other fails at step 0. Undocumented; worth a line in the training CLI help.

---

## BM-06 — Proxy-scale validation

Sign agreement between proxy-scale ablations and the published TabICLv2 full-scale results.
The proxy is not trusted until the signs agree.

| Knob | Published full-scale effect | Proxy steps | Proxy effect | Sign agrees? | Notes |
|---|---|---|---|---|---|
| `--col_ssmax` | | | | | |
| `--col_target_aware` | | | | | |
| `--row_rope_interleaved` | | | | | |

---

## Phase 1 — No retrain

### TP-14 ScoringBench — three hypotheses tested, three negatives

**The premise was probably wrong, and that is the finding.**

I called the published `TabICL v2 (finetuned)` mean rank of 12.05 "an outlier relative to
every other board", "plausibly a harness/config issue rather than a model deficiency", and
"the cheapest potential win in the whole backlog". Having read their code and tested the
three mechanisms that could produce it, **none of them do**. That characterization was too
strong and is withdrawn.

Source: <https://github.com/jonaslandsgesell/ScoringBench>, wrapper at
`scoringbench/univariate/wrappers/tabicl.py`, config at `univariate/config.py`
(seed 42, 5 folds, 3,000-row cap — matching the report). All experiments below use their CV
config and their 200-level alpha grid.

| # | Hypothesis | Mechanism | Result |
|---|---|---|---|
| 1 | Wrapper uses the post-hoc calibrated path | It calls `output_type="quantiles"`; our changelog says `raw_quantiles` are "direct outputs … without post-hoc calibration" | **Identical.** 3 datasets x 2 folds: `wine_quality` 0.2913 vs 0.2913, `boston` 1.1108 vs 1.1118, `kin8nm` 0.0407 vs 0.0407. Under 0.1% apart. |
| 2 | Finetuning degrades calibration | The published entry is the *finetuned* variant (`epochs=80`), not base TabICL | **Neutral.** `boston`, 2 folds: base 1.1108, finetuned 1.1087 (0.2% *better*). |
| 3 | Selection metric mismatched to scoring | `models.py` sets `eval_metric="mse"` — a point metric — while the benchmark scores CRPS | **Neutral.** `boston`, 3 folds: select-on-mse 1.2327, select-on-crps 1.2358 (0.25% *worse*). |

**Caveat, stated plainly:** hypotheses 2 and 3 were tested on `boston` (506 rows) across 2-3
folds. That is underpowered and cannot exclude a small effect. It can exclude the *large*
effect that would be needed to move a mean rank from ~7.6 to 12.05, which is what was being
claimed.

#### What the number probably means

12.05 is a mean rank among **53 methods** on a board densely populated with purpose-built
probabilistic regressors — NGBoost, XGBLSS, BART, normalizing flows, conditional density
estimators, conformal wrappers. TabPFN-3 (7.64) and EXAONE-Tabular (7.59) beat us;
Nori-30M (13.09) does not. Mid-upper-pack on a benchmark that scores *only* the predictive
distribution is a plausible honest result for a model whose headline strength is point
accuracy. It is not obviously an artifact, and I no longer think it is one.

**What would settle it:** running their harness over the real 101 datasets and checking we
reproduce ~12.05. If we do, the item closes as "accurate, and a genuine weakness to work on".
Their dataset list is built lazily from OpenML suites and needs the `openml` package. That is
the remaining work on TP-14, and it is now a verification task, not a bug hunt.

#### One real gap, found and closed

`FinetunedTabICLRegressor` **trains** against pinball loss over the raw quantile head
(`_compute_batch_loss`) but only offered `{"mse", "mae", "r2"}` for early stopping — all point
metrics. Selection and objective disagreed by construction. Added `eval_metric="crps"` so they
can agree, with the CRPS helper pinned against the closed form for a standard normal.

It showed **no measurable benefit** in the one A/B above. It is kept because selecting on the
metric you trained against is principled and the option costs nothing — not because it was
shown to win. Do not cite it as a fix.

### TP-15 fev-bench

| Run | Commit | Ckpt | Harness | SQL skill | MASE skill | Rank /29 | Time vs TabPFN-TS-3.5 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### TP-12 Native date/text preprocessing

| Run | Commit | Encoding | STRABLE Elo | vs 1383 baseline | vs XGBoost-tuned 1336 | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## Phase 2 — Cell encoding and prior

Each item ablated **separately** at proxy scale before any combination is trained.
Prior changes are evaluated on **both** frozen synthetic sets (old prior and new) — a change that
only helps on its own distribution has proven nothing.

| Run | Commit | Item | Scale | Seeds | BM-04 old | BM-04 new | BM-03 | High-card slice | Text slice | Cost delta | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | TP-01 | proxy | | | | | | | | |
| | | TP-02 | proxy | | | | | | | | |
| | | TP-08 | proxy | | | | | | | | |
| | | combined | proxy | | | | | | | | |
| | | combined | full | | | | | | | | |

### TP-11 preprocessing simplification (gated on TP-02)

| Run | Commit | `norm_methods` | BM-03 | Inference time | Reading |
|---|---|---|---|---|---|
| | | `["none","power"]` (baseline) | | | |
| | | `["none"]` | | | |

---

## Phase 3 — Capacity

**TP-05 gate:** cache bytes and single-test-row latency must be flat versus baseline at unchanged
width before TP-06 changes the width at all.

| Run | Commit | Item | Params | Fwd time (s) | Cached 1-row (s) | Cache (MB) | BM-03 | Gate met? | Reading |
|---|---|---|---|---|---|---|---|---|---|
| | | TP-04 | | | | | | n/a | |
| | | TP-05 | | | | | | | |
| | | TP-06 | | | | | | n/a | |

---

## Phase 4 — Multitask checkpoint

Joint training is compared against a **single-task control run at the same proxy scale**, not
against the published claim.

| Run | Commit | Config | Clf metric | Reg metric | Train cost | vs single-task control | Reading |
|---|---|---|---|---|---|---|---|
| | | single-task control | | | | n/a | |
| | | joint (TP-07) | | | | | |

---

## Phase 5 — Research

### TP-09 grouped / non-IID prior

Report both with and without the group identifier exposed as a feature — the deviation from
BeyondArena's standard procedure is TabPFN's to defend and ours to disclose.

| Run | Commit | Prior config | Group id exposed | BeyondArena grouped | BeyondArena temporal | vs tuned+ens MLP | Reading |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## Dropped / did not transfer

Items investigated and abandoned. Kept permanently — this is the record of what does *not* carry
over from TabPFN's model shape to ours.

| Item | Date | Investigated to | Why dropped |
|---|---|---|---|
| | | | |
