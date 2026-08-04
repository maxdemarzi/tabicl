# Performance log

Every measurement worth comparing, with the date it was taken and **what changed since
the previous entry**. `STATUS.md` carries the current numbers; this file carries how they
moved and why.

The reason this exists: several results in this project were compared across runs that
differed in more than one respect — context size, ensemble size, fit set, inference
config, even the pandas version — and each time the difference was attributed to the wrong
cause. A number without its configuration is not a measurement.

**Record for every entry:** date, task, device, `n_estimators`, `use_amp`, context size,
feature spec, fit set, and the one thing that changed.

---

## Standing vs published results (2026-08-04)

RelBench, official protocol. Test ROC-AUC x100. Comparison columns are the published
figures from the TabPFN-3 technical report's Table 14 (arXiv 2605.13986); RelGNN is the
paper's stated SOTA on these tasks and TabPFN-REL its best foundation-model result.

| task | ours | TabPFN-REL | RelGNN | RDBLearn+v3 | vs TabPFN-REL | vs best |
|---|---:|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 80.70 | 79.98 | **85.69** | 82.72 | +0.72 | −4.99 |
| rel-event / user-ignore | 78.11 | 85.38 | **86.18** | 73.70 | −7.27 | −8.07 |
| rel-avito / user-visits | 64.85 | 66.68 | 66.18 | **66.76** | −1.83 | −1.91 |
| rel-trial / study-outcome | 66.50 | **76.43** | 71.24 | 72.89 | −9.93 | −9.93 |

Bold marks the best result per task. **None of them are ours** — we lead TabPFN-REL on
rel-f1 but trail RelGNN there by 5, and trail everywhere else. Ours gets bolded when it
wins a row, not before.

Ours are **calibrated**: every setting chosen on a validation split, test touched once,
AMP off, selection and scoring at the same `n_estimators`. The published figures are
presumably single-configuration, so this is the like-for-like column — an earlier
per-task-maximum table selected on test ran about a point higher and was not comparable.

**Read the gaps honestly.** This is a generic flattening pipeline in front of a stock
TabICL, with no relational machinery in the model and no retraining, against systems built
for relational data. Ahead of TabPFN-REL on rel-f1, within 2 on rel-avito, and well behind
on rel-event and rel-trial.

**Not in the table, and now for a measured reason:** graph-neighbour context beats a random
context of equal size by **+3.10 to +4.94** on rel-event, reproduced across seeds, hop
counts and stratification. It still does not enter the headline, because the calibrated
protocol that would make it eligible **rejects it in 4 of 5 replicates** and returns
82.01 ± 4.94 — a spread eight times the measurement floor. A method only enters the table
if the selection rule picks it, and on this task the selection rule does not. See the
2026-08-04 entry below for why, including a malformed configuration grid that would have
put a *non-graph* result in the table under a graph label.

| selected configuration per task | |
|---|---|
| rel-f1 | `max_columns=None`, no categorical, context 5,000 |
| rel-event | `max_columns=2`, categorical on, context 20,000 |
| rel-avito | `max_columns=None`, no categorical, context 10,000 |
| rel-trial | `max_columns=2`, categorical on, context 10,000 |

**Measurement floor: ±0.6.** Paired-gap sd is 0.29 on rel-event; absolute-score sd is
2.28. Differences under ~0.6 are not resolvable, and unpaired comparisons cannot resolve
five points. Pair everything.

---

## Run log

### 2026-08-04 — calibrated graph-context run COMPLETED: the result stays out, and why
rel-event, L40S at 99 MB/s, AMP off, `n_estimators=4`, 5 replicates of the full protocol
(select on validation, score test once per replicate).

**Environment check passed exactly.** Same runner, same config as the 3090: `+4.80 mean,
sd 3.78 over 3 seeds`, reproducing the recorded cells to the decimal. Numbers from this
host are comparable to earlier ones. *(The full-context cell reads 83.88 here, which is
**not** the 80.93 baseline — that baseline is a different feature build. Checking against
a same-runner cell is what made this conclusive.)*

**Calibrated result: 82.01 ± 4.94 over 5 replicates (range 75.31–88.39), graph chosen 1/5.**
It does not go in the headline table. Three reasons, in order of how badly each would have
misled:

1. **The configuration grid was malformed.** The eligible context pool is 7,111 users
   (val) and 7,184 (test), but the grid offered sizes 2000/5000/**10000** — and
   `select_graph_context` early-returns the *entire pool* once the request exceeds it. So
   "graph, context=10000" was not a graph configuration; it was "use every eligible
   training user". That is the one cell validation ever picked (seed 3, test 88.39). A
   single-replicate run would have reported 88.39 as a graph result. It is not one.
2. **The protocol is too noisy to compare against published single numbers.** Replicate
   spread ±4.94 against a ±0.6 floor. The mean's standard error alone is 2.21.
3. **Validation and test disagree systematically on this method**, not randomly. Graph
   selection scores 74.1–74.5 on validation across all five seeds — stable to ±0.15 —
   while the same selection wins by +4.80 on test.

**Why they disagree is not a signal difference.** The neighbour-label signal is equally
strong on both splits: scoring each query by the positive rate among its labelled train
friends, with no model involved, gives **AUC 73.89 on val and 74.18 on test** (coverage
0.861 / 0.864, median 6 vs 8 train friends). The splits are structurally near-identical.

*Changed since the previous entry:* nothing in the method — this is the same +4.80 effect,
put through a selection rule for the first time.

### 2026-08-04 — class-balance hypothesis for the graph context: refuted
Unstratified reach-ranking builds a **near-single-class context**: positive rate
**0.019–0.046 against a 0.163 base rate**, because ranking by how many queries reach a node
is a popularity ranking and hubs are overwhelmingly negative (selected median degree 26 vs
10 for random). That looked like the explanation.

It is not. Stratifying the selection to preserve the training class balance, as the only
variable moved:

| | A/B gap on test | calibrated |
|---|---:|---|
| unstratified | +4.80 (sd 3.78) | 82.01 ± 4.94, graph 1/5 |
| stratified | +4.94 (sd 3.49) | 82.01 ± 4.94, graph 1/5 |

*The test gap moves by 0.14 — inside the ±0.6 floor, a tie.* The calibrated numbers are
**identical**, because the only configuration validation ever chose was the size-10000
no-op, which never reaches the stratification branch. On validation stratification was
actively worse (62.86 vs 74.43 at size 2000).

*Conclusion:* a 4–8× distortion of the context's class balance changes the result by
nothing measurable. Whatever drives the graph effect, it is not context class balance.
`labels=` is kept in `select_graph_context` because building a 0.02-positive context by
accident is still a defect worth being able to switch off, but it buys no accuracy.

**The finding worth taking forward is the standalone one:** neighbour positive-rate is a
**74 AUC predictor by itself**, no model, one pass over the edge list. That is a feature,
not a context-selection strategy, and it is `RESEARCH.md` item 6b. It needs the negative
controls in `_leakage.py` first — a train row must not see its own label through its
neighbours — and the static-graph caveat still applies.

### 2026-08-04 — graph context confirmed at 8 seeds: +3.10 (1 hop), +3.37 (2 hops)
rel-event, RTX 3090, AMP off, `n_estimators=4`, context 5,000 both arms, paired by seed.
Sanity cell reproduced 80.93 first. Homophily lift +0.2903.

| | mean gap | sd | positive | graph arm | random arm |
|---|---:|---:|---:|---|---|
| 1 hop | **+3.10** | 3.45 | 7/8 | 82.34–85.47 | 75.31–84.43 |
| 2 hops | **+3.37** | 2.76 | 7/8 | 83.66–85.09 | 75.31–84.43 |

*Changed:* seeds 3 → 8, and hops.
*The mean shrank from the 3-seed +4.80 to +3.10*, which is what more seeds usually do and
why three was never enough. Still far clear of the ±0.6 floor. 1 vs 2 hops differ by 0.27,
inside the floor — a tie, so prefer 1 hop as the cheaper.

**The variance result may matter more than the mean.** The graph arm spans 3.1 points
(1 hop) or 1.4 (2 hops); the random arm spans 9.1. Choosing context by graph proximity
makes the score roughly **3× more reproducible**, and nearly all the gap's variance comes
from the random arm's own lottery. The single negative seed (3) is simply where random
drew its best context, 84.43.

*Caveat unchanged:* `user_friends` has no timestamp, so this is an upper bound on a
time-respecting version.

### 2026-08-04 — graph-neighbour context, first look (3 seeds, superseded above)
rel-event / user-ignore, RTX 3090, AMP off, `n_estimators=4`, context 5,000 both arms.
Sanity cell reproduced 80.93 first.

Homophily gate: observed **0.8691** same-label edges against **0.5788** expected under
random assignment, lift **+0.2903** over 37,707 label-known edges. Passed decisively.

| seed | random context | graph context | gap |
|---:|---:|---:|---:|
| 0 | 81.14 | 85.47 | +4.33 |
| 1 | 75.31 | 84.11 | +8.80 |
| 2 | 82.78 | 84.06 | +1.28 |

*Changed:* which rows enter the context. Nothing else — same features, same size, same
seed, paired.
*Mean gap +4.80, sd 3.78, all three seeds positive*, smallest well above the ±0.6 floor.
The graph arm also beats the **full-context** baseline of 80.93 on 5,000 of 19,239 rows.
*Caveats:* three seeds and a wide spread; and `user_friends` carries no timestamp, so the
graph is static and a friendship formed after a prediction time is visible. Upper bound
on a causal version. The random arm's own spread (75.31–82.78) is most of the variance.

*Infrastructure note:* five consecutive L40S hosts had healthy `nvidia-smi` with every
`torch.cuda` allocation failing; a 3090 worked first try. `pod_runner` now tries L40S last.

### 2026-08-04 — calibrated sweep completed
rel-avito **64.85** (was 64.46 hand-picked), rel-event **78.11** (unchanged).
*Changed:* AMP disabled; `select_estimators` now matches `n_estimators`; every setting
chosen on validation. rel-avito picked `max_columns=None` at 10,000 of 116,598 context
rows — 8.6% of the context, which is what takes it off a 46 GB GPU.
*Note:* rel-event's score did not move because the fix changed the selection *regime*, not
the selected *configuration*. It did expose that validation prefers the categorical blocks
(81.63 vs 81.40) where test puts them at −3.52 — genuine val/test disagreement on that
task.

### 2026-08-04 — AMP found to cost 7.3 AUC ⚠ largest single effect in the project
rel-event: CPU **80.93**, CUDA+AMP **73.61**, CUDA−AMP **80.93**.
*Changed:* device and `use_amp` only. Ruled out first: pandas (3.0.5 and 2.3.3 identical),
torch (2.4.1 CPU 80.93 vs 2.12 CPU 80.77), checkpoint (byte-identical).
*Cause:* AMP perturbs predictions by `max|Δp|≈1.8e-02`. Harmless on well-separated data
(synthetic 98.94 vs 98.93), costly where predictions cluster near the boundary.
**Every GPU number taken before this date is suspect.**

### 2026-08-04 — `n_estimators` flips the categorical verdict
rel-event, fit set fixed, AMP off, only `n_estimators` moving:

| `n_estimators` | plain | categorical | gap |
|---:|---:|---:|---:|
| 1 | 75.02 | 77.89 | +2.87 |
| 2 | 79.65 | 78.70 | −0.95 |
| 4 | 80.92 | 77.40 | −3.52 |
| 8 | 81.03 | 77.97 | −3.06 |

*Changed:* nothing but ensemble size. Resolves a long-standing contradiction — both
earlier runners were right in their own regime.
*Mechanism:* plain gains 6 points from ensembling, categorical gains nothing. The extra
columns cost **ensemble diversity**, not feature quality.

### 2026-08-03 — categorical blocks measured across tasks
rel-trial +1.25, rel-f1 0.00 (no categoricals present), rel-event −3.21.
*Changed:* `top_k_categories=4`, `include_mode=True` added.
*Superseded:* the −3.21 was correct but only at `n_estimators=4`; see above.

### 2026-08-03 — context size is task-dependent
rel-avito 5,000 → 63.30, 10,000 → 64.85 (full 116,598 → 64.46).
rel-trial 1,000 → 61.39, 3,000 → 62.66 (full 12,954 → 65.40).
*Changed:* context rows only.
*Finding:* rel-avito's curve is flat, rel-trial's is not. Refutes "10–20k is enough on
every relational dataset". The +0.39 by which rel-avito's 10k beat full context is inside
the noise floor — read it as a tie, which is still the result that matters for cost.

### 2026-08-03 — column budget is a rescue, not a default
`max_columns=2`: rel-event **+3.0** (and the only way it runs at all), rel-trial **0.0**,
rel-f1 **−19.5**.
*Changed:* `max_columns` only.
*Finding:* a 22-point spread from one setting. rel-f1's children are small and entirely
numeric, so capping deletes signal; rel-event's 1,670 features over a 2.5M-row table are
mostly noise.

### 2026-08-03 — relation breadth on rel-trial
3 children 65.05 → 10 children 65.40.
*Changed:* number of child tables.
*Finding:* **+0.35 is inside the noise floor.** Seven extra relations bought nothing
measurable.

### Earlier — row chunking verified exact
`max|Δp| = 1.1e-05`, AUC identical with `offload="auto"` engaged, on both 128- and
161-column feature sets. Survived a direct attempt to break it while hunting the −3.21.

---

## Environment checklist

Run this before trusting any number from a new machine, container or pod. It has caught
two silent failures — the AMP default, and a pod reporting a healthy `nvidia-smi` with
`device_count 1` while every CUDA allocation failed.

1. **Reproduce a known baseline — with the same runner that recorded it.** rel-event plain,
   `max_columns=2`, `n_estimators=4`, full context, AMP off → **80.9**. A near-miss from a
   *different* runner proves nothing: `eval_graph_context` at full context reads 83.88 on a
   verified-good host because it builds different features. Pick a cell the runner in front
   of you actually produced.
2. `torch.cuda` must allocate, not merely report a device.
3. **Measure download bandwidth before scheduling work.** `pod_runner create` now enforces
   a 5 MB/s floor. One host managed 270 kB/s: `nvidia-smi` healthy, CUDA fine, and the run
   simply never started because pip took 25 minutes and a 385 MB dataset never arrived. A
   good host does ~100 MB/s, so the gate costs 60 s and the failure it catches costs a
   session.
4. Record `pandas`, `torch`, device, `use_amp`, checkpoint filename.
5. Never compare across entries that differ in more than one of those.
6. **Check that every configuration in a sweep is the configuration it is labelled as.**
   The graph sweep offered a context size larger than the pool it selects from, so one
   third of the grid was "use everything" wearing a graph label — and that was the cell
   the selection picked.
