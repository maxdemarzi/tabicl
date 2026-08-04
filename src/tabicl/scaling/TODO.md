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

## 1. RESOLVED — the sign does flip with `n_estimators`

Measured on a validated GPU host (AMP off; sanity cell reproduced the CPU baseline 80.92
vs 80.93), fit set fixed at train+val, only `n_estimators` moving:

| `n_estimators` | plain | categorical | gap |
|---:|---:|---:|---:|
| 1 | 75.02 | 77.89 | **+2.87** |
| 2 | 79.65 | 78.70 | -0.95 |
| 4 | 80.92 | 77.40 | **-3.52** |
| 8 | 81.03 | 77.97 | -3.06 |

Both disagreeing runners were right in their own regime. `select_estimators` now defaults
to `n_estimators`; see `DESIGN.md`.

## 2. DONE — calibration fixed

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

## 5. DONE — both numbers re-measured

Calibrated on a verified GPU host, AMP off, selection and scoring at the same
`n_estimators`, sanity cell reproducing the CPU baseline at 80.93 first:

* **rel-event 78.11** — unchanged. The fix altered the selection *regime* but not the
  selected *configuration*, so the score is identical. Worth noting what it exposed: at
  `n_estimators=4` validation still preferred the categorical blocks (81.63 vs 81.40)
  while test puts them at -3.52. Validation and test genuinely disagree on this task, and
  no selection rule fixes that.
* **rel-avito 64.85** — supersedes the AMP-contaminated hand-picked 64.46, chosen on
  10,000 of 116,598 context rows with `max_columns=None`.

The memory guard did its job on the way: it skipped the uncapped spec on rel-event
(predicted 24.4 GB against an 8 GB budget) while allowing it on rel-avito, where it turned
out to be the best choice.

### Remaining

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
