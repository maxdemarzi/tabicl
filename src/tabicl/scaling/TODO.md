# Open work

Written on stopping, so this can be picked up cold. `STATUS.md` is the current state,
`DESIGN.md` the history log. Nothing here is blocking — the branch is committed, tested
(147 passed, 1 skipped) and pushed.

## 1. The open question: does a setting's verdict flip with `n_estimators`?

**This is the one to resume with.** Everything else below depends on the answer.

Two runners disagree about the same configuration on rel-event / user-ignore, official
test split, full context, fit on train+val:

| `n_estimators` | plain `(2, None, False)` | categorical `(2, 4, True)` | gap |
|---:|---:|---:|---:|
| 1 | 80.27 | 81.77 | **+1.50** |
| 4 | 80.77 | 77.84 | **−2.93** |

Ruled out already, so do not re-test:

* **Row chunking.** Cleared directly: `max|Δp| = 1.1e-05`, AUC identical to two decimals,
  with `offload="auto"` engaged, on both a 128-column and a 161-column feature set. The
  exactness claim in `DESIGN.md` holds under adversarial test.
* **Seed noise.** At full context there is no subsampling and results are deterministic —
  five seeds gave identical numbers, sd 0.00.
* **Selection noise on a small validation split.** The categorical blocks won 5/5 seeds
  on validation (+0.39 ± 0.29) and 5/5 on test (+1.26 ± 0.70) at `n_estimators=1`.

`n_estimators` is the only variable left between the two runners.

**Run:** `python -m tabicl.scaling.eval_ensemble_size` — sweeps `{1, 2, 4, 8}` against
both feature specs with everything else fixed. Roughly 45 minutes on CPU; the
`n_estimators=8` cells are ~12 minutes each. It was killed before producing output.

## 2. If the sign does flip: fix the calibration

`eval_relbench_calibrated` selects configurations at `--select-estimators 1` to keep the
sweep affordable, then fits the final model at `--n-estimators 4`. If a setting's sign
depends on ensemble size, that selection is unsound — and it would explain rel-event's
calibrated **78.11** without appealing to noise or distribution shift.

Fix: default `select_estimators` to match `n_estimators`, and make the cheap-selection
shortcut an explicit opt-in with a documented warning. This is a bug in code committed
during this session, and the kind that produces confidently wrong output rather than an
error.

## 3. Then: make `n_estimators` a calibration candidate

If it can flip the sign of another setting, it is a stronger lever than most of what has
been swept here — and unlike the feature knobs it is a standard parameter users already
tune.

## 4. Claims that are provisional until (1) resolves

Both documents carry inline flags at the point of claim; they are not asserting anything
unsupported, but the corrections are unfinished.

* The **−3.21 categorical result on rel-event**, and the conclusion drawn from it that
  the categorical blocks "do not generalise". It reproduces at `n_estimators=4` and
  reverses to **+1.50** at `n_estimators=1`.
* The decision to keep `top_k_categories` / `include_mode` **off by default**. Still the
  conservative choice, but its justification is what is in question.
* The **calibrated rel-event 78.11** in `STATUS.md`, for the reason in (2).

## 5. Unrun: rel-avito calibration

The last empty cell in the calibrated table. It previously reached 24.6 GB on a
5,000-row context fit and drove the machine to 2 MB available, because the as-of scan
sizes its prefix arrays by the *child* table (5.3M and 2.0M rows), not by the context.

`--memory-budget-gb` now predicts and skips such candidates before running them, so it
should be runnable. Start conservatively and watch memory.

## Notes for whoever resumes

* **Pair every comparison.** Same seed, one variable changed. Measured on rel-event:
  absolute-score sd 2.28 (range 5.45) against paired-gap sd 0.29 — pairing tightens the
  estimate ~8x. Unpaired, nothing under about five points is resolvable here.
* **Differences under roughly ±0.6 are not measurable** on these tasks. Two claims were
  downgraded to ties on that basis (relation breadth +0.35, rel-avito 10k context +0.39).
* **Vary one thing at a time.** Every wrong conclusion in this session came from two
  runners differing in more than one respect — context size, ensemble size, fit set, and
  inference config all drifted between scripts.
* **`TaskStop` kills the shell, not its child Python.** A "stopped" job kept holding
  ~20 GB. Check for orphaned processes after killing anything.
* **Verify edits landed.** A `str.replace` that does not match fails silently; asserting
  is not enough if the shell continues to the next command anyway. Prefer the editing
  tool, which fails loudly.
