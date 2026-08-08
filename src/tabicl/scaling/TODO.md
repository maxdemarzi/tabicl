# Open work

Written on stopping, so this can be picked up cold. `STATUS.md` is the current state,
`DESIGN.md` the history log. Nothing here is blocking — the branch is committed, tested
(176 passed, 1 skipped) and pushed.

## STATE AS OF 2026-08-08

**The features are not the constraint; the selection rule is, and it now has a partial fix.**

* All four traversal shapes are built (`--depth2`, `--siblings`, `--dimensions`). Each is
  real on a fixed configuration — t of 2.8 to 4.8 — and worth roughly nothing after
  selection. Two produce the identical signature on the same task, helping every arm except
  the one validation picks.
* **`--drop-stale-arms` is the first intervention here to survive its own controls**: it
  refuses arms whose feature block's coverage collapses between validation and test, using
  links and timestamps only. Two gains (+0.69, +0.53 on our two worst tasks), five
  bit-identical results including all four control tasks, no losses. **Awaiting a
  12-replicate confirmation** — five estimates in that session shrank on more replicates and
  this one is not exempt.
* **Measure features on FIXED-configuration arms, not the calibrated block.** Fixed arms
  correlate at r = 0.88–0.94 across seeds and pairing is worth 2.4–3.0× on the standard
  error; calibrated arms correlate at −0.03 to +0.34 and pairing buys nothing, because
  selection varies per seed. `tabicl.scaling.paired` reports `r` and warns below 0.5.
* Refuted with measurements, do not rebuild: label history (per-entity), abstention,
  entity novelty, novelty matching, entity time deltas, ensembling over configurations
  (twice, the second time properly powered).

## DECISIONS WAITING ON THE MAINTAINER (2026-08-07)

Four items where the measurement is done and the call is someone else's. Each says what the
evidence is and what it does *not* settle.

1. **Align the library's `Table.max_columns` to 4.** The runner default was changed on
   validation across four tasks by worst-case regret: 4 is never more than 2.13 off the best
   value for a task, against 19.13 for the old hardcoded 2 and 6.12 for the library's
   `None`. **The library still ships `None`**, so `flatten_relational` called directly does
   not match the runner. Three places defined a default and disagreed; two now agree.

2. **DONE — categories is now on by default (`--categories 8`).** It was held back because
   calibrated test showed **−0.44** on rel-event/user-ignore against validation's +0.26. That
   number came from six replicates at sd 1.43, so its standard error was about the size of
   the effect. Re-run at **twelve**, with the reading fixed beforehand: **+0.15**, and
   rel-trial +0.09, user-repeat +0.93. All positive, none negative — though only user-repeat
   clears the ±0.6 floor, so the other two are best read as *no effect* rather than as small
   gains. The baseline itself moved 81.98 → 81.54 between replicate counts, which is the
   honest measure of that task's noise. **Reversible with `--categories 0`**, and every
   standing number predates it.

3. **The TabICL `feature_mask` bug, documented and not fixed.** `predict_proba` builds
   `feature_mask` over the input feature space and indexes it against a filter fitted in a
   reduced one. Reachable only when a test column is entirely NaN. `allow_nan = True`
   advertises support that path does not deliver.

4. **`eval_backbone.py` needs a `TABPFN_TOKEN`.** The runner is written and tested; `tabpfn`
   ≥ 6.0.0 will not download weights without a registered priorlabs.ai account with the
   licence accepted. The 2.x line is ungated but is a generation below what every peer
   method uses. Until then, the nearest evidence is RDBLearn's own three rows: a backbone
   generation on fixed features is worth **+0.72 average, sd 3.16**, against our 2.60
   deficit — and it is *negative* on the task where our peer gap is largest.

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
* **Terminate, and verify nothing is left running.** This reverses the advice that stood
  here, which was to `stop` so the volume survives a re-download. It was wrong on cost: a
  stopped pod still bills for its volume, and *every* cycle since has had to hunt for a
  usable host anyway — the probe now rejects hosts for broken CUDA, for torch below 2.2,
  and for bandwidth under 5 MB/s, so the preserved volume is rarely the one you come back
  to. `cycle.ps1` terminates in a `finally` and prints `REMAINING PODS:` afterwards, so a
  pod cannot outlive its job even if nothing is watching.

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
* **Probe the host before trusting it, on three counts.** All three failures are silent:
  a broken community host reports a healthy `nvidia-smi` while every allocation raises; a
  host shipping torch below 2.2 imports and allocates perfectly and then raises inside the
  model's forward pass (multi-dim `Tensor.all`), which cost a full cycle after
  provisioning, upload, install and a 385 MB download had all succeeded; and a slow host
  looks entirely healthy and simply never finishes. `pod_runner.probe_host` checks CUDA
  from torch, the torch version, and download bandwidth, in that order.

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
* ~~The decision to keep `top_k_categories` / `include_mode` **off by default**.~~
  **RESOLVED 2026-08-05, and neither of the earlier numbers was right.** Both blocks turn
  out never to have run in `eval_track_record` at all — it sets `windows` and
  `max_columns` and nothing else — so every standing number was measured without them and
  the ±3-point swings above came from `eval_relbench_calibrated`, whose own bundling made
  them unreadable: `(2, 4, True)` was its only categorical cell, entangling the blocks
  with `max_columns=2`, and it factorized train and test separately besides. Gated
  properly, one variable at a time: rel-trial +0.27 / −0.09, rel-event +0.01 / −0.39,
  rel-avito +0.05, and on rel-f1 and rel-avito the variants **refuse to run** because
  those schemas have no eligible categorical child columns. Every cell inside the ±0.6
  floor. Off by default is correct, and now for a measured reason.
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
