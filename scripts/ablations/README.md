# Ablation queue

The runs to launch once a GPU is available, in the order the plan wants them. Each is
`scripts/train_proxy.sh` with one `EXTRA` flag set — everything else is pinned to Stage 1.

## Phase 0 — validate the proxy first (BM-06)

These are not ablations, they are the calibration. Each knob below has a known full-scale
effect from the TabICLv2 paper; the proxy is only trusted once the **sign** of each agrees.
If a sign disagrees, raise `STEPS` and repeat.

```bash
SEED=42 bash scripts/train_proxy.sh                                          # control
ABLATION=ssmax_off   EXTRA="--col_ssmax False --icl_ssmax False" bash scripts/train_proxy.sh
ABLATION=notarget    EXTRA="--col_target_aware False"            bash scripts/train_proxy.sh
ABLATION=rope_interl EXTRA="--row_rope_interleaved True"         bash scripts/train_proxy.sh
```

## Phase 2 — cell encoding and prior

```bash
ABLATION=tp01 EXTRA="--col_fourier_value True"                   bash scripts/train_proxy.sh
# TP-02 (ECDF features) and TP-08 (high-cardinality prior) are not implemented yet.
```

## Phase 3 — capacity

TP-04 first, because it is the stability prerequisite for the width change and must be shown
at-worst-neutral at the *current* width before anything relies on it.

```bash
ABLATION=tp04 EXTRA="--qk_norm True"                             bash scripts/train_proxy.sh
# TP-05 (GQA) and TP-06 (width) are not implemented yet. TP-06 must not run before TP-05
# holds the KV cache flat -- see benchmarks/RESULTS.md for the measured 48 KiB/row/estimator.
```

Run each at `SEED=42,43,44`. The control must be re-run on the **same card** as the
treatment — `pod_doctor.json` records which, and a treatment on an H100 against a control on
an A100 is a hardware difference, not an ablation.

## Reading the result

```bash
python -m benchmarks.report --ledger benchmarks/_results/real_small.jsonl \
    --metric accuracy --against control
python -m benchmarks.report --ledger benchmarks/_results/real_small.jsonl \
    --metric log_loss --lower-is-better --against control
```

Check `log_loss` as well as `accuracy`. Several backlog items are expected to move
calibration more than they move accuracy, and accuracy alone would call those a wash.
