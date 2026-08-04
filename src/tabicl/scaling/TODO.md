# Open work

Written on stopping, so this can be picked up cold. `STATUS.md` is the current state,
`DESIGN.md` the history log. Nothing here is blocking — the branch is committed, tested
(147 passed, 1 skipped) and pushed.

## 0. AMP contaminates every GPU number — re-measure before comparing

`use_amp=True` is the default in all three inference configs and costs **7.3 AUC** on
rel-event (CPU 80.93, CUDA-with-AMP 73.61, CUDA-without-AMP 80.93). See `DESIGN.md`.

Consequences for the items below:

* **Any GPU measurement in these documents predates this finding**, including rel-avito's
  64.46. Re-measure with `use_amp=False` before quoting.
* **GPU is now the right place for this work.** With AMP off a rel-event fit takes ~20s on
  a 24 GB card against ~250s on CPU, and it reproduces CPU *exactly*. That converts
  five-seed paired comparisons from an hour into a minute, which is the thing that would
  have prevented most of the wrong conclusions in `DESIGN.md`.
* **`pod.py stop`, not `terminate`.** Terminating destroys the volume, and re-downloading
  rel-event costs ~30 minutes at the ~700 kB/s these pods get. Stop preserves it.

Set it like this:

```python
NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
TabICLClassifier(device="cuda:0", inference_config=NOAMP)
```

### Run remotely, always

**Standing rule: no heavy compute on the local machine.** Model fits, sweeps and
benchmarks go to a remote GPU (RunPod; `pod.py` in the session scratchpad handles
create/exec/stop, token at `~/.runpod/token.txt`).

Two reasons, and the second matters more. Local runs repeatedly destabilised the
workstation — four incidents in one session took available memory to 235 MB, 2 MB, 854 MB
and near-zero, because TabICL fits here routinely want 20–30 GB and `TaskStop` kills the
shell without killing its child Python. And a rel-event fit is ~250s on CPU against ~20s
on a GPU, so multi-seed paired comparisons cost an hour and claims kept resting on one
seed — which is the root cause of most wrong conclusions in `DESIGN.md`.

### Pod hygiene, learned the hard way

* **Always run a sanity cell first.** Reproduce a known CPU number before trusting any new
  environment. This is what caught AMP, and what caught a second pod whose CUDA was
  broken -- both of which had produced confident, plausible, wrong output.
* **Community pods can be silently broken.** One L40S host reported a healthy
  `nvidia-smi` and `device_count 1`, yet `torch.cuda.is_available()` was False and every
  allocation failed with "CUDA unknown error", with `CUDA_VISIBLE_DEVICES` unset or set.
  `/dev/nvidiactl` was present. Not fixable from inside; terminate and take another host.
* **`stop` preserves the volume, `terminate` destroys it.** Re-downloading rel-event costs
  ~30 minutes at the ~700 kB/s these pods get, so stop unless the host is bad.

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

**Correction, and this reopens the question.** `n_estimators` was *not* the only variable:
the +1.50 runner fit on train only (19,239 rows) and the -2.93 runner on train+val
(21,252). Two variables, which is the error this project keeps repeating. And the first
GPU attempt at the sweep confidently reported "the sign FLIPS" — an artifact of AMP, since
every cell in it was distorted.

So the question is still open and must be re-run with **one** variable moving and
`use_amp=False`: identical fit set, identical features, `n_estimators` in {1, 2, 4, 8}.
`eval_ensemble_size` already holds the fit set fixed at train+val, so it is the right
harness; it needs the no-AMP config added.

**Run:** `python -m tabicl.scaling.eval_ensemble_size` — sweeps `{1, 2, 4, 8}` against
both feature specs with everything else fixed. Run it on a GPU pod with AMP disabled (the harness does this automatically when
`TABICL_DEVICE` is not `cpu`); each cell is seconds there. No result has been produced yet.

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
