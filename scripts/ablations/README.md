# Ablation queue

The runs to launch once a GPU is available, in the order the plan wants them. Each is
`scripts/train_proxy.sh` with one `EXTRA` flag set — everything else is pinned to Stage 1.

## Phase 0 — validate the proxy first (BM-06)

**Revised 2026-09-19 after reading TabICLv2 §7 / Appendix C. Do not run the old version.**

These are not ablations, they are the calibration. Each knob below has a *published*
~100-Elo / ~64%-win-rate effect in the TabICLv2 paper; the proxy is trusted only once it
reproduces the **sign** of each.

**Gate: evaluate on `cc18_narrow` (62 datasets), not `real_small`.** On 8 informative datasets
a true 64% effect clears p < 0.05 only ~15% of the time — the calibration would be unable to
tell a working proxy from noise. See `benchmarks/RESULTS.md`.

```bash
SEED=42 bash scripts/train_proxy.sh                                            # control
ABLATION=no_target EXTRA="--col_target_aware False"                bash scripts/train_proxy.sh
ABLATION=no_ssmax  EXTRA="--col_ssmax False --icl_ssmax False"     bash scripts/train_proxy.sh
ABLATION=adamw     EXTRA="--muon False --lr 1e-4"                  bash scripts/train_proxy.sh
```

Expected sign for all three: **worse than control.**

- **`adamw` must use `--lr 1e-4`**, per the paper's own AdamW ablation, not Muon's 8e-4.
  An AdamW run at a Muon learning rate would lose for the wrong reason.
- **RoPE is not in this list.** An earlier version included `--row_rope_interleaved` as a knob
  "with a known full-scale effect". RoPE interleaving is never ablated in the paper; that
  reference did not exist.
- The paper measured these at **280K steps** and notes per-step noise falls as the learning
  rate decays. Save checkpoints often and read the *curve*, not one endpoint — the question
  is at what step the arms separate, if they do at all.

Evaluate every checkpoint, with `--model-path` (without it the suite measures the *released*
model and the whole calibration is silently meaningless):

```bash
python -m benchmarks.suites.real_small --suite cc18_narrow --run-id BM-06 \
    --config-id no_target-step5000 --model-path /workspace/ckpt/proxy-no_target-seed42/step-5000.ckpt \
    --device cuda
```

Read it with a paired test on log-loss, not a sign test on accuracy.

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

## Phase 4 — multitask checkpoint (TP-07)

Three arms rather than two — a joint checkpoint has to match a single-task control on *each*
task — so it has its own launcher. It scores the joint arm on `cc18_narrow` and `ctr23`, each
control on its own suite.

```bash
SEED=42 bash scripts/tp07_run.sh          # clf_control, reg_control, joint on gpus 0,1,2
# only if joint loses on either task:
SHARED_EXTRA="--qk_norm True --input_norm True" RUN_ID=TP-07-tp04 bash scripts/tp07_run.sh
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
