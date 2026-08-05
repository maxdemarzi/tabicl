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
| rel-f1 / driver-top3 | 82.48 | 79.98 | **85.69** | 82.72 | +2.50 | −3.21 |
| rel-event / user-ignore | 78.11 | 85.38 | **86.18** | 73.70 | −7.27 | −8.07 |
| rel-avito / user-visits | 64.85 | 66.68 | 66.18 | **66.76** | −1.83 | −1.91 |
| rel-trial / study-outcome | 69.36 | **76.43** | 71.24 | 72.89 | −7.07 | −7.07 |

Bold marks the best result per task. **None of them are ours** — we lead TabPFN-REL on
rel-f1 but trail RelGNN there by 5, and trail everywhere else. Ours gets bolded when it
wins a row, not before. rel-trial moved 66.50 → 69.36 on 2026-08-04 and is still last on
that task; a narrowed gap is not a win.

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
| rel-trial | `max_columns=2`, **shared-key track record on**, context 12,000 |

**Measurement floor: ±0.6.** Paired-gap sd is 0.29 on rel-event; absolute-score sd is
2.28. Differences under ~0.6 are not resolvable, and unpaired comparisons cannot resolve
five points. Pair everything.

---

## Run log

### 2026-08-04 — rel-f1 80.70 → 82.48, and the cause is a column budget, not a feature
rel-f1 / driver-top3, L40S, AMP off, 5 calibrated replicates, **all five link tables carry
timestamps** so `--timed-links-only` changes nothing and the result is causal throughout.

| configuration | calibrated test | val |
|---|---:|---:|
| `max_columns=2` (the hardcoded default) | 69.90 ± 2.86 | 68.99 |
| **`max_columns=None`** | **82.48 ± 0.80** | 87.82 |

**+12.58 from one setting**, and the A/B says the track-record features are almost
incidental to it: `base` alone scores **82.16**, with `+struct` adding +0.72 (sd 1.34,
4/5) — barely over the floor. Control 4 passes.

*This is a lever we already knew about and were mis-applying.* "Column budget is a rescue,
not a default" has been in this file since 2026-08-03, recording −19.5 on rel-f1. The
runners then hardcoded `max_columns=2` for every task anyway, so every rel-f1 number they
produced was measured on a base crippled by a setting we had already measured as harmful
there. The finding was written down and not wired in.

*Caveat, same as rel-trial's.* The previous 80.70 came from `eval_relbench_calibrated`'s
config space, this from `eval_track_record`'s. Both are calibrated protocols choosing on
validation and touching test once, so "our best calibrated rel-f1 is now 82.48" is
defensible; it is not a like-for-like ablation. **Still not a win** — it passes TabPFN-REL
(79.98) and trails RDBLearn (82.72) and RelGNN (85.69).

### 2026-08-04 — rel-event, resolved: the 89.48 was untimed links. Case closed.
Dropping the two link tables that carry no timestamp (`user_friends`, both directions) and
keeping only `event_interest` and `event_attendees`:

| configuration | calibrated test | validation chose |
|---|---:|---|
| all link tables (untimed included) | 89.48 ± 0.67 | `+struct` 4/5 |
| **timed link tables only** | **81.53 ± 3.01** | **`base` 3/5**, `+struct` 2/5 |

**−7.95, and the selection rule stops choosing the feature at all.** The A/B agrees:
`+struct over base` is **+1.16 (sd 1.82, 2/3 seeds)**, inside the floor. So the whole
apparent gain was counting friendships that formed *after* the prediction time. Nothing
about graph structure on rel-event is reportable, and 78.11 stands.

*Why the controls did not catch this on their own, which is the transferable part.* Control
4 (temporal on `n_linked`) **passed** — because an untimed table's degree does not change
when the cutoff is shifted, so it contributes nothing for a temporal control to detect. The
control was not wrong, it was **blind**, and a blind control passing looks exactly like a
clean one. The fix is structural rather than statistical: `--timed-links-only` excludes
what cannot be tested. *A control can only clear a feature whose value it can make move.*

### 2026-08-04 — rel-trial base sweep: nothing clears the floor
Validation only, 3 seeds, one knob at a time from a fixed reference
(`max_columns=2`, 3 children, 30/365-day windows, `n_estimators=4`, val 61.74).

| knob | best | Δ |
|---|---|---:|
| `max_columns` | 2 / 4 / 8 / None **all 61.74** | 0.00 |
| `n_estimators` | 1 → 61.77 | +0.04 |
| child tables | 10 → 62.28 | +0.54 |
| windows | 365/1095 → 61.92 | +0.19 |

**Best overall +0.54, under the ±0.6 floor — a tie.** The base pipeline is not being held
back by any of these settings on rel-trial, so the gap to a published entity-only baseline
is not a tuning problem.

*One genuine finding:* `max_columns` is a **no-op on this task** — identical to four
decimal places at every value, 134 features throughout — because rel-trial's child tables
have at most two usable columns anyway. The earlier "column budget: 0.0 on rel-trial" entry
was recording that the knob does nothing here, not that the budget is harmless.

### 2026-08-04 — entity-only baseline, second attempt: still not reproduced. Stop guessing.
Fixed the two obvious defects — datetimes now become *age at the cutoff* rather than being
dropped, and near-unique object columns (`forename`, `surname`, `url`) are excluded instead
of factorised into arbitrary row ids. It improved and still does not reproduce:

| task | GBDT-entity, attempt 1 | attempt 2 | published | still short by |
|---|---:|---:|---:|---:|
| rel-f1 | 36.43 | 42.49 | 73.92 | **31.4** |
| rel-avito | 50.68 | 50.68 | 53.05 | 2.4 |
| rel-trial | OOM | 59.61 | 70.09 | **10.5** |

**Two attempts is enough to stop guessing at their feature construction.** The remaining
gap is too large to be tuning; their "entity features" almost certainly are not what this
harness builds from `db.table_dict[entity]`. Either read RelBench's own baseline code or
drop the comparison — do not publish a third guess, and do not quote our +30 over it.

*What the run does establish*, one variable apart inside one harness:

| task | relations are worth | our model over GBDT, same features |
|---|---:|---:|
| rel-f1 | **+19.80** | +10.41 |
| rel-avito | **+14.40** | +1.08 |
| rel-trial | **+1.74** | +1.11 |

The flattening layer pays for itself — and **barely at all on rel-trial (+1.74)**, which is
exactly the task where a published entity-only baseline is said to beat us. That is
consistent with the relational features being near-useless there rather than with our
model being weak, and it is why the base sweep matters more than another feature family.

### 2026-08-04 — entity-only baseline: harness does not reproduce it yet, do not quote
Attempt to reproduce Rel-LLM's LightGBM-on-the-entity-table-alone baseline. Three arms so
"relations do not help" and "our model is behind" stay separable.

| task | GBDT-entity (ours) | GBDT-entity (published) | TabICL-entity | TabICL-full |
|---|---:|---:|---:|---:|
| rel-f1 / driver-top3 | **36.43** | 73.92 | 47.40 | 70.34 |
| rel-avito / user-visits | **50.68** | 53.05 | 51.77 | 66.17 |
| rel-trial / study-outcome | — OOM — | 70.09 | — | — |

**The reproduction failed and the bolded column is the evidence.** 36.43 is *below chance*
— that is a broken arm, not a weak baseline. `entity_only` keeps non-datetime, non-key
columns of the entity table and factorize-encodes categoricals, which leaves **5 features
on rel-f1 and 4 on rel-avito**. Their LightGBM plainly sees more. So the comparison that
prompted all this — their 70.09 on rel-trial against our 69.36 — **still has not been
made**, and nothing here licenses "we beat LightGBM".

*What the run does support*, since the two TabICL arms differ in one variable and share a
harness: **relational features are worth +22.94 on rel-f1 and +14.40 on rel-avito** over
the same model on entity columns alone. That is a genuine ablation of the flattening layer
and the first direct evidence it pays for itself — but measured against a weak entity arm,
so read it as "relations help", not as a magnitude.

*Two operational faults.* rel-trial OOMed because **an orphaned process from an earlier run
still held 32.95 GiB** — the known failure where killing a shell leaves child Python alive;
check for orphans before scheduling GPU work. And an inline `ssh` command with nested
quotes failed to parse for the second time today; remote work goes in a script file.

### 2026-08-04 — rel-event 89.48 WITHDRAWN: the temporal control fails
The calibrated rel-event number reached **89.48 ± 0.67**, chosen by validation 4/5, which
would have beaten every published result (RelGNN 86.18, TabPFN-REL 85.38, ICL+MLP 84.02).
**It is withdrawn. Do not quote it.**

Two defects, found by fixing the first one:

**1. Structure from the future.** The winning arm was `+struct`, built on `n_linked` — a
count of entities sharing a key — and the first implementation used the link table whole,
with no time filter. A friendship or event signup formed *after* the prediction cutoff was
counted. None of the three controls could see this: they move label cutoffs, and this
column consults no labels. `key_target_history` now takes `link_times` and does an as-of
degree count. rel-event has two timestamped link tables (`event_interest`,
`event_attendees`) and two without (`user_friends`, both directions), so that task is only
partly fixable — the runner now prints which is which and warns that untimed keys give an
upper bound.

**2. With that fixed, the temporal control fails outright:**

```
control 2: *** LEAK *** withholding history *improved* the score to 0.8240 from 0.8181
```

Earlier it passed at *exactly* the tolerance boundary (0.8231 against 0.8231), which was
already recorded here as "not a clean pass". It is now over. The runner refuses to report
a lift, which is the behaviour it was built for.

*Read the two together.* The number moved 85.34 → 89.48 as features were added, and the
control moved from vacuous, to boundary, to failing. The lift and the leak grew together.
That is the signature this whole control apparatus exists to catch, and it caught it one
step before the number reached the headline table.

**What is still open, and it is narrow.** `+struct` uses only `n_linked`, while control 2
tests the *rate* columns which that arm does not contain. Control 3, on the count columns,
passes. So a structure-only result may yet survive — but it needs a temporal control on
`n_linked` specifically, and until that exists the arm is untested rather than vindicated.
Nothing goes in the table on the strength of "the failing control tested a different
column".

### 2026-08-04 — rel-event track record: 85.34 calibrated, PROVISIONAL, not in the table
rel-event / user-ignore, L40S, AMP off, 5 paired seeds then 5 calibrated replicates.
**Calibrated 85.34 ± 1.48 (range 83.38–86.98) against a standing 78.11**, which would sit
level with TabPFN-REL (85.38) and just under RelGNN (86.18). It is **not** in the headline
table, for a reason that is nobody's fault but mine:

**The controls tested the wrong arm.** Both the permutation and temporal controls were run
on the `positive_rate` columns, and validation chose the **counts-only** arm 5/5. The
reported number therefore rests on an uncontrolled feature. A permutation test is
meaningless for counts — they do not depend on label *values*, so shuffling changes
nothing — which means the missing check is specifically a temporal control on the count
columns. Until that runs, this stays out.

Transfer gate first, which is why only rel-event was built:

| task | best key | coverage | standalone test AUC | vs our calibrated | built? |
|---|---|---:|---:|---|---|
| rel-avito | LocationID | 0.892 | 58.90 | below 64.85 | no |
| rel-f1 | constructorId | 0.971 | 72.72 | below 80.70 | no |
| rel-event | event co-attendance | 0.956 | **82.32** | above 78.11 | yes |

A/B, base features identical: **+4.70 over base** (sd 1.60, 5/5) but **only +1.33 over
counts-only** (sd 0.72, 5/5). So unlike rel-trial — where outcome history beat counts by
+4.59 — most of rel-event's gain is *connectivity*, not outcome content. Same degree
phenomenon that made the graph-context result look better than it was, caught this time by
the arm that exists to catch it.

**Two control defects found and fixed, both mine.** The temporal control originally used
hardcoded 180/365-day shifts. rel-event's horizon is 7 days and its span 147, so the shift
removed every usable label, the feature went constant, and the control "passed" at exactly
0.5000 having tested nothing. Shifts are now a fraction of the task's own span, and
coverage is printed at each so a vacuous pass is visible. With that fixed the control
passes **at exactly the tolerance boundary** — withholding 22 days *improved* the score by
0.005 against a 0.005 tolerance. That is not a clean pass either, and it is a second reason
this number is provisional.

### 2026-08-04 — shared-key track record on rel-trial: +5.45, and it enters the table
rel-trial / study-outcome, L40S, AMP off, `n_estimators=4`, 5 paired seeds, base features
identical in every arm. **The first result in this project to change the headline table:
rel-trial 66.50 → 69.36.**

The feature: outcome history among trials sharing a sponsor, condition, facility or
intervention, as of this trial's cutoff. A flattener cannot produce it — it aggregates a
related row's *columns*, never its *outcome*, and the outcome carries the base rate.

| arm | mean gap | sd | positive |
|---|---:|---:|---|
| history over base | **+5.45** | 0.30 | 5/5 |
| history over counts-only | **+4.59** | 0.81 | 5/5 |

*Changed:* 15 columns added to an unchanged pipeline. Nothing else.
*sd 0.30 against a ±0.6 floor* — 9× the floor and the most reproducible effect measured
here. Compare graph context at +3.10 with sd 3.45.

**The counts-only arm is why this is believable.** On rel-event an apparently strong
neighbour-label feature turned out to be mostly *degree* — a count with no outcome
information at all. So the count columns got their own arm, and history beats them by
+4.59. The outcome content is doing the work, not the connectivity.

**Calibrated, and selected rather than assumed: 69.36 ± 0.98 over 5 replicates (range
67.66–70.15). Validation chose `+history` 5 times out of 5**, at context 6,000–12,000.
That unanimity is what item 8 never achieved, and it is the difference between an effect
and a reportable number.

*Gate first, as it should have been for item 8.* Each key was measured standalone before
any machinery was built: condition 60.4, sponsor 61.4, facility 60.6, intervention 61.4
test AUC at 44–88% coverage. Three near-independent keys, each nearly as strong alone as
the entire pipeline was.

*Controls:* permutation clears its null by 6.4 sd; earlier cutoffs produce **exactly** no
improvement (0.6560 at every shift). The 365-day resolution horizon matters more here than
anywhere else in the project — a trial's outcome takes a year to resolve, so ignoring it
would let every row read a year of the future.

**Caveat on the comparison, stated because the two numbers come from different pipelines.**
The previous 66.50 was calibrated over `max_columns` and categorical blocks; this 69.36 is
calibrated over arm and context size with 365/1095-day windows. Its base arm scores 63.82,
*below* the old 66.50 — so the +5.45 is the controlled measurement and the headline change
is "our best calibrated number on this task", not a like-for-like ablation. Adding the
track record to the stronger base has not been measured and should be.

### 2026-08-04 — label propagation as a feature: controlled, and it does not help
rel-event, L40S, AMP off, 5 paired seeds, context 10,000, base features identical in every
arm. Three arms, because "propagation helps" and "label content helps" are different
claims and only a degree arm separates them.

| `n_estimators` | propagation − base | propagation − degree-only |
|---:|---:|---:|
| 4 | **−0.74** (sd 1.61, 1/5 positive) | +0.93 (sd 1.24, 4/5) |
| 8 | **−0.38** (sd 1.54, 2/5 positive) | +0.96 (sd 1.49, 3/5) |

*Changed:* three feature columns added to an unchanged pipeline. Nothing else.
*Result:* no gain at either ensemble size, both inside the noise floor of zero. Ensemble
diversity — which flipped the categorical verdict — does not rescue it: −0.74 and −0.38
are a tie with each other as well as with zero. Label content does beat degree-only by
~+0.95 consistently, so within the block the labels are doing something; the block as a
whole still adds nothing the relational features were not already supplying.

**Two corrections to the number this experiment was launched on.** The "74 AUC standalone
neighbour-label signal" recorded earlier today was wrong twice over:

- **It was mostly degree.** `labelled_degree` alone — no label content at all — scores
  **73.24**. The positive-rate feature scores **67.78**, *below* it.
- **It was reading unresolved labels.** The original 74.18 used every training label
  regardless of time. A neighbour's outcome resolves over the 7 days *after* its own
  prediction time; restricting to labels resolved at the query's cutoff costs **6.4 AUC**.
  That gap is the measure of how much a plausible-looking temporal feature can borrow from
  the future.

**The controls earned their place, in both directions.** The permutation control *failed*
first (permuted score 0.5354 against a 0.02 tolerance). It was not a leak: a test query's
own label is never in the label set, so self-leak is impossible by construction, and what
survives permutation is degree. The fix was the correct null — `permutation_test` against
the permuted distribution, which the real labels clear by **6.4 sd** — not a loosened
tolerance. `permutation_control` remains right for features whose entire content is labels
and wrong for graph features, which encode structure too.

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
