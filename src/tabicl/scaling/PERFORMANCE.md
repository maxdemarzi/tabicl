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

| task | ours | DFS † | TabPFN-REL | RelGNN | RDBLearn+v3 | vs TabPFN-REL | vs best |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 81.98 | 76.81 | 79.98 | **85.69** | 82.72 | +2.00 | −3.71 |
| rel-event / user-ignore | 80.98 | 77.95 | 85.38 | **86.18** | 73.70 | −4.40 | −5.20 |
| rel-avito / user-visits | 65.54 | 65.81 | 66.68 | 66.18 | **66.76** | −1.14 | −1.22 |
| rel-trial / study-outcome | 72.26 | 69.12 | **76.43** | 71.24 | 72.89 | −4.17 | −4.17 |

† **DFS is measured here, not published.** Deep Feature Synthesis (Featuretools) run over the
same tables with per-row cutoff times and scored by **the same TabICL, same context, same
seeds** — only the feature builder differs. Every other column is a published figure from
another system with its own model, so DFS is the one entry that isolates *our aggregation*
from *our model*. It is the baseline the phrase "a generic flattening pipeline" has been
implicitly claiming parity with since this file began; `RESEARCH.md` 6e.

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

### 2026-08-05 — which child tables: dict order spends a slot on a causally-empty table

`--children N` takes the first N child tables in dictionary order. Two findings, one of
which invalidates the comparison the other is trying to make.

**Dict order is not stable across machines.** rel-trial's first three are
`['designs', 'eligibilities', 'drop_withdrawals']` on the pod and
`['conditions_studies', 'designs', 'drop_withdrawals']` on the workstation — same ten
candidates, same code, different three used. So `--children 3` names a different feature
set on different hosts, and two runs of the "same" configuration are not comparable. This
is the storage-order defect in its most consequential form: earlier instances picked the
wrong rows or the wrong columns, this one picks a different experiment.

**`--top-children` ranks them on validation instead**, univariately (no fit required) and
corrected for width by scoring each block against three permuted targets — a maximum over
k columns grows with k under pure noise, so an uncorrected rule ranks tables by how wide
they are. rel-trial:

| child table | margin | best column | permuted null | cols |
|---|---:|---:|---:|---:|
| facilities_studies | **+4.5** | 7.3 | 2.8 | 39 |
| sponsors_studies | +2.8 | 5.5 | 2.7 | 40 |
| designs | +2.6 | 4.1 | 1.4 | 33 |
| eligibilities | +2.6 | 4.1 | 1.4 | 33 |
| conditions_studies | +1.7 | 3.9 | 2.2 | 39 |
| drop_withdrawals | +0.0 | **0.0** | 0.0 | 41 |
| outcomes | +0.0 | **0.0** | 0.0 | 30 |
| reported_event_totals | +0.0 | **0.0** | 0.0 | 41 |
| outcome_analyses | +0.0 | **0.0** | 0.0 | 52 |
| interventions_studies | −0.0 | 3.8 | 3.8 | 39 |

**Four tables yield no usable column at all** — best AUC exactly 50, every column dropped
as all-NaN or constant after the cutoff. Two of them are `outcomes` and `outcome_analyses`,
independently proven 100% post-cutoff in the depth-2 entry; the ranking found that without
being told. `interventions_studies` is the width case the correction exists for: best 3.8
against a null of 3.8, so all of it is the maximum-over-39-columns effect.

**And the dict-order default spends one of its three slots on `drop_withdrawals`,** one of
the four empty ones.

Calibrated, 3 seeds: dict-order **72.24 ± 1.14** (VAL 68.56), validation-ranked
**73.32 ± 0.41** (VAL 68.50).

**Not claimed.** Test says +1.08 with a 3× tighter spread; validation says −0.06 against an
SE near 0.45, so the selection instrument cannot distinguish them and "test improved" is
precisely the reasoning that has produced three retractions in this log. Re-running at 5
seeds to see whether validation resolves it. The reproducibility argument for
`--top-children` stands on its own regardless of the outcome, since the thing it replaces
is not well-defined.

### 2026-08-05 — context temporal locality: large on rel-event, absent everywhere else

The context is drawn uniformly at random from train. Nobody chose that, and the selection
thread had already diagnosed the train→test gap as temporal. `recent` takes the most recent
rows by timestamp instead. Paired by seed, identical features both arms, 5 seeds:

| task | train span | train→test gap | recency @ ctx 10,000 | recency @ ctx 1,000 |
|---|---:|---:|---:|---:|
| **rel-event** | 147 d | 15 d | +0.31 (SE 0.57) | **+7.50** (SE 0.96, 5/5) |
| rel-avito | 8 d | 10 d | +0.90 (SE 0.34) | −1.50 (SE 0.97) |
| rel-trial | 6,570 d | 731 d | +0.99 (SE 0.28) | −1.42 (SE 1.04) |
| rel-f1 | 3,870 d | 1,976 d | not measurable¹ | −0.33 (SE 0.50) |

¹ context 10,000 covers all 1,353 training rows, so `recent` selects the same set as
`random`; the runner skips it rather than report the noise.

**The mechanism is temporal locality, and it is available on exactly one task.** Recency
pays when the recent slice sits much closer to the test period than an average training
row. rel-event's last 1,000 rows cover roughly the final week of a 147-day span with test
15 days later. Everywhere else the gap dwarfs the span — 731 days on rel-trial, 1,976 on
rel-f1 — and rel-avito's entire training set is 8 days wide, so there is nothing to buy and
the narrower context only costs diversity. rel-avito @1,000 was a *prediction* of this
account before it was measured, and it came back −1.50.

**Two explanations were tried and discarded first**, both worth recording because each
looked convincing at the time:

* *"Recency is a general property of temporal splits."* The @10,000 column alone supports
  it — +0.90 and +0.99, two tasks, same direction. The @1,000 column reverses both signs.
* *"It is a class-balance effect."* At context 10,000 the context whose positive rate sits
  nearer the test prior wins on all four tasks, which is a 4/4 match. At context 1,000 the
  two largest results both contradict it: rel-event's recent slice is *farther* from the
  test prior (0.0412 vs 0.0328) and gains 7.50, and rel-trial's is much *closer* (0.0105 vs
  0.0555) and loses 1.42.

**rel-event `recent` @1,000 scores 86.83 / 86.81 / 87.03 / 86.58 / 86.59** — mean 86.77,
sd 0.18 — against our standing **80.98** and RelGNN's published **86.18**. The tight spread
is expected, not suspicious: `recent` is deterministic, so all five seeds share one context
and only the model varies. It beats the standing *configuration* by 5.8, so it is not an
artefact of comparing against a weak small-context baseline.

**This is test-side and is not in the table.** It enters only if the calibrated protocol
selects it on validation. One change was needed before that is even possible: the
calibrated grid runs `cap//4, cap//2, cap` = 2,500 / 5,000 / 10,000, so it could never have
tried the setting that produces the effect — it would have measured the diluted +0.31 and
concluded recency does nothing. The grid now extends to 1,000 whenever more than one
context order is offered.

**Promoted, and rejected: validation chose `random` 3 times out of 3** on rel-event, taking
a large random context every replicate. So recency joins graph-neighbour context,
resampling, child count and per-key selection on the list of effects that are real on test
and not selectable — the fifth.

*(Two flaws in that run, both mine. It omitted `--timed-links-only`, so its 78.72 includes
rel-event's two **untimed** `user_friends` link tables and the `+struct` arm it selected is
an upper bound rather than a causal result — not comparable to the standing 80.98. And the
output filter dropped the per-configuration `val=` lines, leaving no record of how far
behind the recency configurations scored. Both fixed for the re-run.)*

*The same caveat does **not** apply to the other tasks, which I initially claimed it did:
all four of rel-avito's link tables carry timestamps and all five of rel-trial's do, so
`--timed-links-only` is a no-op on both. rel-event is the only task in the benchmark where
the flag changes anything.*

**On rel-avito the calibrated protocol *chose* recency, 3/3** (`recent`, `recent-half`,
`recent`, all at context 10,000, arm `+struct`), giving **65.62 ± 0.16** against a standing
**65.54 ± 0.11**. That is a legitimate calibrated result and the first time a recency
setting has been selected — and it is +0.08, comfortably inside the ±0.6 floor, so it is a
tie rather than a gain. The gate's +0.90 did not survive calibration, which is the usual
fate of a gate result and the reason the gate is not the table.

**All three promotions, and the shape of the answer:**

| task | order validation chose | calibrated test | standing | delta |
|---|---|---:|---:|---:|
| rel-event | `random` 3/3 | 78.72 ± 1.61 ¹ | 80.98 | — ¹ |
| rel-avito | **recency 3/3** | 65.62 ± 0.16 | 65.54 ± 0.11 | +0.08 |
| rel-trial | **recency 2/3** | 71.72 ± 1.57 | 72.26 | −0.54 |

¹ Not comparable: this run omitted `--timed-links-only` and rel-event is the one task where
that matters, so its `+struct` arm includes two untimed `user_friends` tables.

**Recency is selectable on two tasks of three, and worth nothing on either.** Both deltas
sit inside the ±0.6 floor. The one task where it is worth a great deal — rel-event, +7.50
on test — is the one task where validation rejects it. Offering it as an option cost 0.54
on rel-trial, which is what a free parameter does when it has nothing to find.

**Why validation rejects it is structural, and measurable.** The train→validation gap is
smaller than the train→test gap on every task in the benchmark:

| task | train span | train→val | train→test | ratio |
|---|---:|---:|---:|---:|
| rel-event | 147 d | 7 d | 15 d | 2.1× |
| rel-avito | 8 d | 4 d | 10 d | 2.5× |
| rel-trial | 6,570 d | 365 d | 731 d | 2.0× |
| rel-f1 | 3,870 d | 150 d | 1,976 d | **13.2×** |

A setting chosen on validation is tuned for a *shorter horizon* than the one it is scored
at. Anything whose value grows with distance from the training period is therefore
systematically undervalued, and recency is exactly such a setting — which is why a +7.50
test effect is rejected unanimously.

`--gap-validation` carves a pseudo-validation split out of train whose distance from its
own fitting pool matches the train→test gap. No extra feature build (both are subsets of
the same matrix), and the final fit still uses all of train and touches test once; only the
selection geometry changes. Constructible on rel-event (11,522 pool rows, 2,013 pseudo-val)
and rel-trial (8,620 / 960), and **not** on rel-avito or rel-f1, where the gap is too large
a fraction of the training span to leave a pool at all.

This reopens the selection thread closed on 2026-08-05, and the distinction matters: that
verdict came from criteria scored on held-out *train* rows with no gap whatsoever — nearer
to the fitting pool than the real validation split. The defect here is a specific measured
mismatch between the selection horizon and the scoring horizon, not another fold
arrangement. If gap-matched selection also picks `random`, recency is genuinely
unselectable and that is the answer.

### 2026-08-05 — two feature blocks that had never run in production, now measured and dead

`eval_track_record` builds every child `Table` with `windows` and `max_columns` and nothing
else. It had never set `top_k_categories` or `include_mode`, so `_category_histogram` and
`_prefix_mode` — both implemented, both unit-tested, both documented as carrying signal the
numeric path cannot — contributed **nothing to any number in the standing table**. Wired up
and gated (paired by seed, one variable, test-side, 5 seeds, `max_columns=2`, 3 children):

| task | variant | gap | sd | SE | positive | columns added |
|---|---|---:|---:|---:|:---:|---:|
| rel-trial | categories | +0.27 | 0.42 | 0.19 | 4/5 | 60 |
| rel-trial | mode | −0.09 | 0.44 | 0.20 | 2/5 | 6 |
| rel-avito | mode | +0.05 | 0.15 | 0.07 | 3/5 | 1 |
| rel-event | categories | +0.01 | 1.42 | 0.63 | 2/5 | 30 |
| rel-event | mode | −0.39 | 1.47 | 0.66 | 2/5 | 3 |

**Every cell is inside the ±0.6 floor** — but the two tasks are not equally informative
and the sd column alone hides that. On rel-trial and rel-avito the standard errors are
0.07–0.20, so those are tight nulls: an effect of half a point would have shown. On
rel-event, SE 0.63 means five seeds could only have resolved a gap of roughly 1.3 or more,
so the honest reading there is "no effect detected by an underpowered test", not "no
effect". rel-event's paired variance has been the widest of the four throughout.

Taken together the blocks do not pay, and on the tasks where the measurement is sharp that
is a real answer rather than a shrug. The runner now prints SE alongside sd so this
distinction is not left to whoever reads the table.

On **rel-f1 and rel-avito every categorical variant refused to run at all** — the runner
now exits when a requested block emits zero columns rather than scoring identical frames.
Their child tables are entirely numeric, confirmed on rel-f1 with the budget removed
(428 features, still nothing eligible). "Not applicable here" and "measured at no effect"
are different findings and used to be indistinguishable in this log.

**Two bugs this exposed, both of which produced a clean-looking null:**

1. **`eval_depth2` was passing the target in as a feature.** `flatten_relational` forwards
   every entity column except the key and the cutoff, and the task table carries the
   target. It scored **AUC 100.00 at both depths** on rel-trial, where the standing number
   is 72.26 — and the *difference* was +0.00 with sd 0.00 over five seeds, which is exactly
   what a careful null looks like. The other three callers drop the target; this one never
   did. `_guards.assert_no_perfect_feature` now refuses any frame where a single column
   separates the target (≥99.5), wired into all three runners. It tests the symptom, so it
   catches a renamed target, a duplicate, or an aggregate that reconstructs one.

2. **`numeric_booleans` was a no-op on the entire benchmark.** It keyed off
   `is_bool_dtype`, and RelBench spells every boolean `'t'`/`'f'` in an **object** column —
   `eligibilities.adult`, `designs.subject_masked`, `studies.is_fda_regulated_drug` and
   eight more on rel-trial alone. Flags are now recognised by their values. Those eleven
   columns currently contribute one `nunique` of 1 or 2 each; their *rate* has never been
   computed. Unmeasured as of this entry.

### 2026-08-05 — depth-2 cannot be measured on this benchmark, and now for a proven reason

Not "measured at no effect". Not available, on all four tasks, for three different reasons:

| task | timed children | with timestamped grandchildren | blocker |
|---|---:|---:|---|
| rel-f1 | 3 | **0** | the schema has no depth-2 to build |
| rel-event | 3 | 1 | entity keys repeat → ambiguous-cutoff guard |
| rel-avito | 3 | 1 | entity keys repeat → ambiguous-cutoff guard |
| rel-trial | 10 | 1 | **the whole subtree postdates the cutoff** |

rel-trial was the one task where depth-2 was reachable, and it is empty:

* **0 of 158,246** `outcome_analyses` rows linked to a training study precede that study's
  deadline. Minimum lag **1 day**, median 195, maximum **365** — which is exactly
  `task.timedelta`, the label horizon.
* **0 of 117,592** `outcomes` rows do either.

So `outcomes → outcome_analyses` *is* the outcome being predicted, and a correct cutoff
empties it completely. That also explains the old 49.12 → 49.37: those features were built
from rows that are the label, the leak was total, and it still produced chance — which says
the figure was never informative in either direction.

**Three separate defects had to be cleared to reach that answer**, and each produced
`+0.00, sd 0.00` — output indistinguishable from a careful null:

1. `max_columns` deleted the grandchild block outright (nested statistics ranked against
   raw columns on coverage; a grandchild block is sparse, so depth-2 emitted output
   byte-identical to depth-1). Fixed → 88 → 305 features.
2. The **target was passed in as a feature**, so both arms scored AUC 100.00.
3. With both fixed, 15 columns appear and every one is entirely NaN, so the model drops
   them and the predictions match to two decimals across five seeds.

The runner now refuses each of the three rather than reporting a number for it. The third
refusal is the one worth keeping: *columns can be present and still carry nothing*, and
"+0.00 over five seeds" is what that looks like from the outside.

**Per-row nesting is not worth building.** It would unblock rel-event and rel-avito, but
the only evidence about whether depth-2 pays comes from the one task where it was
reachable, and there the honest answer is that the question is unanswerable rather than
answered.

### 2026-08-05 — selection machinery closed out: CV cannot judge resampling, at any fold scheme
Forward-chaining folds were built because random k-folds train on the future to predict the
past. They fail identically:

| criterion | resample=1 | resample=3 |
|---|---:|---:|
| CV, random folds | 90.37 | **95.67** |
| CV, time-ordered | 86.76 | **94.23** |
| **test** | **81.69** | 80.25 |

**The fold scheme was never the problem.** Both criteria gain ~7.5 points on resampling
while test falls 1.44. The cause is structural: resampling averages over draws from the
*training pool*, which improves in-distribution coverage substantially and transfers poorly
to a later period. **Any criterion scored on held-out train rows will overvalue it**, so no
arrangement of those rows fixes this.

**And resampling's gain was arm-dependent.** With the arm held at `base`, resampling *hurts*
on test: 81.69 → 80.25. The earlier +1.26 came with `+struct` selected. So "averaging over
context draws reduces variance and helps" was too general a claim — it interacts with which
features are present, which is not what a pure variance-reduction argument predicts and is
reason enough to distrust the mechanism story I attached to it.

**Child count, re-measured under the same instrument, shrank — which vindicates declining it.**

| children | time-ordered CV | test |
|---:|---:|---:|
| 3 | **67.33** | 71.97 |
| 6 | 67.16 | **72.27** |

CV ranks 3 first by 0.17 (noise), the same ordering validation gave. But **the test gap fell
from +0.87 to +0.30**, now inside the floor. A real +0.87 should have persisted across a
change of selection regime; one that halves is more consistent with noise that happened to
favour 6 in the first draw. So "measured but unselectable" was too generous for this one:
the better reading is *probably smaller than it looked, and declining it was right*.

**Both threads are closed.** Resampling is not adopted, and no further selection machinery
is worth building. Three effects remain recorded as measured-but-unselectable — per-key
selection (+0.67) and resampling on two arms — with the caveat this entry supplies: an
effect that only one instrument can see, and that shrinks when another looks, should be
treated as probably-noise rather than as value trapped behind a bad instrument. That is the
opposite of the conclusion I was drifting toward an hour ago.

### 2026-08-05 — context resampling: works on test, unadoptable, and CV made it worse
Averaging predictions over independent context draws. Test improves on **3/3** tasks and the
spread collapses on **3/3** — rel-event 80.16 → 81.42 with sd 1.98 → 0.43, rel-trial 72.32 →
72.71, rel-f1 +0.28. That is the mechanism's own prediction, not a pattern found afterwards.

**It is not adopted, because no valid selection rule picks it.** Validation prefers
`resample=1` on 2 of 3 tasks (the differences are 0.16–0.40, noise). So `--cv-folds` was
built to give a better-powered criterion — and it failed in the most informative way:

| criterion | resample=1 | resample=3 |
|---|---:|---:|
| CV score | 90.37 | **95.67** |
| test | **81.01** | 80.25 |
| arm chosen | `+struct` | `base` |

**CV got 5.3 points more confident while test fell 0.76 and the selected arm flipped.**
The cause is structural and I should have seen it before building: random k-folds over
train break the temporal ordering the whole benchmark is built on — a random fold trains on
the future to predict the past, so a variance-reduction trick that helps *within* a period
is rewarded far beyond what it earns on a later one. A criterion that grows more confident
as test degrades is worse than a noisy one.

**Where that leaves it.** Resampling is real on test, costs no feature columns, and cannot
be claimed. The honest statement is not "resampling fails" but **"no selection instrument
available here resolves a 0.3–1.3 effect"** — validation is too small (588–2,013 rows, and
it has now failed to see a real effect four separate times), and k-fold CV is biased on a
temporal split. The correct version is a *time-ordered* split of train, which is the obvious
next thing and was not what I built.

### 2026-08-05 — DFS measured at last: we beat it on rel-f1, tie on rel-trial
`RESEARCH` 6e, after seven failed attempts. Same TabICL, same context, same seeds — only the
feature builder differs, and DFS was given the *tuned* primitive set (`num_unique`, `mode`,
`trend`, `time_since_last`, `avg_time_between`, `skew`, plus date transforms) and per-row
cutoff times so it respects the same temporal boundary we do.

| task | DFS | ours | ours − DFS | feature build |
|---|---:|---:|---:|---|
| rel-f1 / driver-top3 | 76.81 | 81.84 | **+5.03** (sd 1.49, 5/5) | DFS 28 s / 193 cols · ours **1 s** / 428 |
| rel-event / user-ignore | 77.95 | 80.34 | **+2.39** (sd 3.16, 4/5) | DFS 119 s / 811 cols · ours **25 s** / 2000 |
| rel-trial / study-outcome | 69.12 | 69.56 | +0.45 (sd 0.50, 5/5) | DFS 20 s / 116 cols · ours **2 s** / 134 |
| rel-avito / user-visits | 65.81 | 65.51 | **−0.30** (sd 0.52, 2/5) | DFS **414 s** / 94 cols · ours **18 s** / 176 |

**Complete: we lead 2 of 4, tie 2.** rel-f1 +5.03 and rel-event +2.39 are real; rel-trial
+0.45 and rel-avito −0.30 are both inside the floor. **rel-avito is the one task where DFS
is nominally ahead**, and it is the task where our own pipeline is closest to the published
state of the art — so the aggregation is not what limits us there.

The build-time gap is the durable result: **1 s / 25 s / 2 s / 18 s against 28 s / 119 s /
20 s / 414 s.** On rel-avito that is 23×, and DFS spent seven minutes producing 94 columns
against our 176 in eighteen seconds. That is the O(n log n) prefix scan against per-cutoff
recomputation, and it is why "generic flattening pipeline" understates the layer even where
the accuracy ties.

**The claim these documents have leaned on all along now has evidence.** "A generic
flattening pipeline in front of a stock TabICL" was *too modest* on rel-f1, where our
aggregation beats the standard automated approach by 5.03 with every seed agreeing, and
about right on rel-trial, where +0.45 is inside the floor — a tie. And it builds features
**20–28× faster**, which is the O(n log n) prefix scan against per-cutoff recomputation.

*Read the caveats.* Both arms are capped at 3 child tables (`kids[:3]` still lives in that
runner), so this is DFS-on-3 against ours-on-3, not against our full layer. DFS also lost
some column types to a woodwork-safety cast, so it is a competent rather than maximal
configuration. And rel-event and rel-avito were never run.

*Cost of getting here, recorded because it was disproportionate:* seven attempts. Three
genuine library incompatibilities (woodwork vs pandas 3, the frame-naming API, pandas
`Categorical`), **two self-inflicted** (`--device cpu` on a GPU host; `--no-deps` on a
third-party install stripping woodwork), and two provisioning failures. The diagnostic
finally cost more than several of the experiments it was meant to contextualise.

### 2026-08-05 — child-table cap: a point available on test, declined on validation
Every runner sliced `kids[:3]` — three child tables by dictionary order. **Only rel-trial is
affected**: the other three have exactly 3 child tables, and removing the cap reproduced
rel-f1 at 81.98 and rel-event at 80.16 to the decimal.

| children | **val** | test |
|---:|---:|---:|
| **3 (kept)** | **68.65** | 72.32 ± 0.94 |
| 6 | 68.59 | **73.19 ± 0.42** |
| 8 | 68.16 | 72.33 ± 0.97 |
| all 10 | 68.04 | 72.91 ± 0.74 |

**73.19 would have put rel-trial above RDBLearn (72.89). It is declined.** Validation ranks
3 first and 6 second, separated by 0.06 — noise — so the honest procedure keeps 3 and the
0.87 on test is only reachable by looking at the answer. This is the second time today a
near-table-changing number has been available by test-selection, after rel-avito's 66.21.

*The more useful observation is that validation spans 0.61 across all four settings.* It has
essentially **no power to choose child count on this task** — 960 validation rows cannot
resolve differences this size. So the right conclusion is not "3 is better than 6" but
"this decision cannot be made on our validation split", and 3 is kept because it is the
incumbent and the cheapest, not because it won.

*The cap was still worth removing:* `--children` is configurable now, and used-vs-available
prints on every run, so the next task with ten child tables will not silently use three.

### 2026-08-05 — settled at 12 replicates: rel-event's drop was noise, rel-trial's gain is real
Both post-fix numbers were 5-replicate figures compared against 12-replicate predecessors,
which is a two-variable comparison — a replicate count is a variable.

| task | pre-fix | post-fix, 5 reps | post-fix, **12 reps** | verdict |
|---|---:|---:|---:|---|
| rel-event | 81.76 (12) | 80.16 ± 1.98 | **80.98 ± 2.07** | −0.78 at ~1.3 SE — **not significant** |
| rel-trial | 71.50 (8) | 72.32 ± 0.94 | **72.26 ± 0.64** | +0.76 at ~4 SE — **real** |

**rel-event's −1.60 was noise**, and would have gone into the log as a regression caused by
the encoder fix. Its replicates span 76.67–85.46 — nearly nine points — so five of them
cannot resolve a 1.6 difference, and the honest conclusion is that fixing the categorical
codes left rel-event unchanged.

**rel-trial's gain holds** and the spread tightens to 0.64. At 72.26 it is **0.63 from
RDBLearn (72.89)** and clear of RelGNN.

*On text, a softer correction than the last entry.* Across 12 replicates rel-trial's
validation chose `+rate` 9 times, `+text+rate` 2 and `+text` once. So text is not worthless
post-fix — it is selected occasionally — but it is no longer the default choice, and the
previous entry's "not selected at all" was itself a 5-replicate over-reading. The measured
position: **text is a marginal, sometimes-selected feature on rel-trial, not the +2.14
headline it appeared to be.**

### 2026-08-05 — categorical codes were inconsistent across splits; all four re-measured
`_numeric` called `pd.factorize` on each frame independently, and factorize codes by order
of first appearance — so **the same category got different integers in train and test**. Not
merely arbitrarily ordered but *inconsistently* ordered: the model learned a mapping that did
not hold where it was applied. Categories are now fitted on train and reused, unseen values
encoding to −1. Every number in the table was previously taken through this.

| task | broken codes | fixed | Δ |
|---|---:|---:|---:|
| rel-f1 | 82.48 | 81.98 ± 0.87 | −0.50 (tie) |
| **rel-trial** | 71.50 | **72.32 ± 0.94** | **+0.82** |
| rel-event | 81.76 | 80.16 ± 1.98 | −1.60 |
| rel-avito | 65.61 | 65.54 ± 0.11 | −0.07 (tie) |

**rel-trial gains 0.82 and is now 0.57 from RDBLearn (72.89)**, comfortably past RelGNN.

**The interesting part is what it did to selection.** rel-trial's validation now chooses
`+rate` **5/5**, having chosen `+text+rate` 8/8 before. Fixing the encoding made the plain
relational features good enough that **text no longer earns its place** — so today's text
result was partly compensating for a defect elsewhere in the pipeline. The +2.14 attributed
to text on 2026-08-05 should be read as "+2.14 given a broken categorical encoder", and the
honest current statement is that text is not selected on rel-trial at all.

*Two caveats, both against my own numbers.* rel-event's −1.60 compares 5 replicates against
12, and its sd is 1.98, so that gap is about 1.8 standard errors and not established;
it needs 12 replicates before the drop is believed. And every "gain" recorded earlier today
was measured through the broken encoder, so the *sizes* of those gains are not reliable even
where their direction was.

### 2026-08-05 — text representation size: 32 is right, and the encoder question is closed
rel-trial, calibrated, only the SVD component count moving.

| components | val | test |
|---:|---:|---:|
| 16 | 67.33 | 71.27 ± 0.84 |
| **32** | **67.88** | **71.50 ± 0.38** |
| 64 | 67.41 | 70.91 ± 0.80 |
| 128 | 67.15 | 68.92 ± 0.85 |

**Validation and test agree on 32**, so the standing configuration was right and is now
chosen rather than assumed. 16/32/64 span 0.59 — at the floor — while 128 loses 2.6.

**This closes the sentence-encoder question without building one.** More representational
capacity does not help here; past 64 it actively hurts. A transformer embedding supplies
precisely what is already in surplus, so the expected gain is negative, and the plan to try
`all-MiniLM-L6-v2` is dropped rather than deferred. Gate-before-building applied to our own
pipeline — one cycle instead of a model download, a new code path and a day.

*The 128 drop is the ensemble-diversity effect again* — third sighting today, after the
categorical blocks and the `n_estimators=16` selection flip. A wider feature block costs
something independent of whether the extra columns carry signal. That is now a reliable
enough pattern to predict with, not merely to observe.

*Infrastructure:* host blacklisting worked first time — one CUDA rejection, then a usable
host, against eight consecutive failures on the same machine in the previous cycle.

### 2026-08-05 — child text lifts rel-trial to 71.50, past RelGNN, and as-of filtering proves itself
Child-table text aggregated **as of each row's cutoff**, 8 calibrated replicates.

**71.50 ± 0.38, `+text+rate` chosen 8/8** — up from 70.26, and past **RelGNN's 71.24**. The
first published system this project has overtaken on any task. Still short of RDBLearn
(72.89) and TabPFN-REL (76.43), so rel-trial is not won.

A/B, base features identical: base 63.82, `+counts` 64.68, `+rate` 69.27, `+text` **69.68**
(was 68.25 with entity text only), `+text+rate` **71.41**. Child text adds +1.43 to the text
arm, and text and outcome history remain complementary.

**The as-of filter did visible work, and this is the part worth keeping.** Of 26 candidate
child text columns, only `designs.*` and `eligibilities.*` survived. Every `outcomes.*` and
`outcome_analyses.*` column **emptied out entirely** — a trial's outcome text is written
after it finishes, so at prediction time none of it exists.

Those are exactly the columns the naive gate scored highest: `outcomes.title` at 63.23,
`outcomes.description` at 61.85. Reading them would have meant predicting a trial's outcome
from the write-up of its outcome. The gate had no way to see it, because it concatenated
without regard to time; the as-of filter removed them without anyone having to notice. That
is the structural-over-procedural principle working as intended — the leak was made
impossible rather than tested for.

### 2026-08-05 — rel-trial at 12 replicates: 70.26 ± 1.38, and a child column is being missed
Twelve replicates give **70.26 ± 1.38** against 70.36 ± 1.74 on five — consistent and
tighter. Validation chose `+text+rate` 7/12 and `+rate` 5/12. The entry holds at 70.26.

**The repaired child-table gate found a column we are not using.** After fixing the
all-NaN mapping, the surviving child columns score honestly, and one is as strong as the
best entity column:

| column | source | coverage | test AUC |
|---|---|---:|---:|
| `official_title` | entity | 0.995 | 64.10 |
| **`eligibilities.criteria`** | **child** | **1.000** | **63.95** |
| `detailed_descriptions` | entity | 0.621 | 62.88 |
| `brief_title` | entity | 1.000 | 60.77 |

`eligibilities.criteria` is full-coverage, as strong as anything in the entity table, and
**not in the `+text` arm**, which embeds entity columns only. Extending the text block to
child tables is the obvious next step — with as-of aggregation, since the gate concatenates
without regard to time and its number is therefore an upper bound.

*The gate fix earned itself twice over.* The phantom columns are gone (they were the
zero-coverage ones), and the remaining scores are plausible rather than suspiciously tidy.

**rel-avito and rel-f1 confirmed to have no text columns at all**, entity or child, so 6f
cannot help them and that door is closed on a working gate rather than a broken one.

### 2026-08-05 — text pays off on rel-trial: 69.36 → 70.36, and it is complementary
TF-IDF + SVD over the entity table's free-text columns, vectoriser fitted on **train only**,
as its own arm. rel-trial, 5 seeds, base features identical across arms.

| arm | A/B mean |
|---|---:|
| base | 63.82 |
| `+counts` | 64.68 |
| `+text` | 68.25 |
| `+rate` (outcome history) | 69.27 |
| **`+text+rate`** | **71.70** |

**Text and outcome history are complementary, not redundant** — the question the four-arm
design existed to answer. Text alone is worth **+4.43** over base, and still **+2.43** on
top of the outcome history. Two different kinds of signal, and the pipeline had neither
until today.

**Calibrated: 70.36 ± 1.74**, validation choosing `+text+rate` 3/5 and `+rate` 2/5 — a
**+1.00** gain over 69.36, above the ±0.6 floor. rel-trial is now 0.88 from RelGNN's 71.24.

### 2026-08-05 — rel-event: the gate said no, running it anyway made things worse
rel-event's single text column scored **67.74 standalone against an 81.76 pipeline** — a
14-point deficit, where rel-trial's best column was 5.3 below its pipeline. The gate rule
is "near or above"; this was neither, and it was run regardless.

**Offering the text arm on rel-event produced a calibrated 79.56 ± 1.04 against 81.76
without it, with validation choosing `+text` 5/5** on a validation score of 88.39 against a
test of 79.56. Adding an arm to the config space made the protocol *worse*, because
validation on 2,013 rows preferred it and test did not.

**rel-event keeps 81.76 and text is not offered there.** That decision rests on the gate,
which rejected the column before any of this ran — not on having seen the test number,
which would be exactly the test-selection this file keeps catching. The useful lesson is
that the gate is not merely a time-saver: **an arm that fails it can actively cost you**, by
being available for validation to pick.

### 2026-08-05 — text gate: rel-trial is worth building, and my rel-f1 claim was wrong
TF-IDF + SVD + logistic per text column, standalone, no GPU.

| task | text columns | best column (test) | ours |
|---|---:|---|---:|
| rel-trial | 8 | `official_title` **64.10**, `detailed_descriptions` 62.88, `brief_title` 60.77, `brief_summaries` 60.20 | 69.36 |
| rel-event | 1 | `location` 67.74 | 81.76 |
| rel-f1 | **0** | — | 82.48 |
| rel-avito | **0** | — | 65.61 |

**Correction to the previous entry.** I wrote that text embedding explains the 31-point
rel-f1 gap against RelBench's LightGBM baseline. **It does not: rel-f1's entity table has no
free-text columns.** Its seven columns are ids, short names and a date. Whatever their
baseline does better there, it is torch_frame's *categorical* stype handling — proper
categorical encoding rather than my factorised integers — not embeddings. The same applies
to rel-avito, also zero text columns. The general claim "we ignore text" stands; the
specific attribution of rel-f1's gap to it was wrong and is withdrawn.

**rel-trial is worth building.** Four columns score 60.2–64.1 standalone against a pipeline
at 69.36. That ratio is the same one that justified the shared-key track record — its keys
gave 60–61 against a 66.50 pipeline and went on to add **+5.45** — and unlike those keys,
text is a genuinely different kind of signal rather than more aggregation of the same rows.

*Two cautions carried into the build.* `official_title` and `brief_title` are near-unique
(11,829 and 11,933 distinct over 11,994 rows), so they sit close to the identifier trap the
gate is meant to exclude; they pass only because their tokens repeat even when the strings
do not, and a leak here would look like a strong result. And `biospec_retention` failed
outright — 11 features against 64 SVD components — which is a reminder to fit the encoder to
the column rather than apply one setting everywhere.

### 2026-08-05 — why the LightGBM baseline could not be reproduced: it embeds text
Read `examples/lightgbm_entity.py` in the RelBench repo rather than guessing a third time.
The baseline merges task and entity tables exactly as we do, then builds a
`torch_frame.data.Dataset` with `col_to_stype` from `get_stype_proposal` **and a
`TextEmbedderConfig`**. Free-text columns are embedded.

**Ours discards them.** `_numeric()` factorises non-numeric columns into arbitrary integers,
and `eval_entity_baseline` excludes near-unique object columns as identifiers — correct if
the alternative is factorising them into row ids, wrong if the alternative is embedding.
That is the 31-point rel-f1 gap, and two reproduction attempts could never have closed it.

**The earlier framing was wrong and is withdrawn.** "A LightGBM on the entity table alone
beats our whole relational pipeline" is not the finding. The finding is **"text embeddings
beat a pipeline that ignores text"** — a different claim, and an actionable one. Every entry
above that leans on the entity-only comparison should be read with this correction.

*This is now the largest untried lever*, and it points at our worst task: rel-trial is
−7.07, its studies carry descriptions, eligibility criteria and intervention text, and
TabPFN-REL reaches 76.43 there. See `RESEARCH.md` 6f.

### 2026-08-05 — per-key selection: no gain, and a near-miss worth recording
`RESEARCH` item 12. Every key's block was concatenated indiscriminately, so this ranks keys
by standalone validation AUC and keeps the top *k*. Calibrated, 5 replicates.

| task | keys kept | **val** | test |
|---|---|---:|---:|
| rel-avito | top 1 | 69.52 | **66.21 ± 0.53** |
| rel-avito | top 2 | 70.78 | 65.51 ± 0.16 |
| rel-avito | **all 4** | **77.37** | 65.54 ± 0.11 |
| rel-trial | top 1 | 64.86 | 65.24 ± 1.79 |
| rel-trial | top 2 | 65.38 | 66.91 ± 0.88 |
| rel-trial | top 3 | 65.97 | 68.85 ± 1.12 |
| rel-trial | **all 5** | **66.77** | 69.27 ± 0.86 |

**No gain, and the reason is the whole point.** On rel-avito, top-1 scores **66.21** — better
than all-keys, and 0.55 from RDBLearn's 66.76, the closest this project has come to a win.
It is not ours to claim: **validation ranks the subsets in the opposite order** (77.37 for
all keys against 69.52 for top-1), so the honest procedure keeps all four and returns 65.54.
The 66.21 is reachable only by reading test.

Had `k` been picked the way it is tempting to pick it — run all three, quote the best — this
file would now claim rel-avito 66.21 and a near-win. That is the same test-selection that
inflated the retired per-task-maximum table by about a point, arriving by a new route.

On rel-trial both columns rise monotonically with *k* and agree, so validation keeps all
five keys and the entry is unchanged at 69.36. Selection is consistent there; it just has
nothing to add.

*What it does establish:* concatenating every key is already the right call on both tasks,
and the weak keys are not diluting the strong ones the way the categorical-blocks result
suggested they might. That hypothesis is now measured rather than assumed.

### 2026-08-05 — rel-event at 12 replicates: 81.76 ± 2.95, and the entry firms up
The standing rel-event number carried sd 3.01 over 5 replicates, making its gain over the
old 78.11 about two standard errors — the weakest thing in the table. Twelve replicates,
`max_columns=2`, timed links only:

**81.76 ± 2.95, range 75.31–86.11, `+struct` chosen 10/12.** Standard error 0.85, so the
+3.65 over 78.11 is **~4.3 SE** rather than ~2. The entry survives scrutiny.

*But read the range, not the mean.* Individual replicates span **10.8 points**, from 75.31
to 86.11 — the same replicate lottery that made the original graph-context result look
better than it was. The mean over 12 is trustworthy; any single run of this configuration
is not, and a 5-replicate version of this number could have landed anywhere in a 3-point
band. This is why the table quotes replicate means and why 5 was too few.

*The `max_columns=None` arm did not finish* even at 5400 s — it builds 2000 features on
19,239 rows. The two settings were a tie at 5 replicates (81.22 against 80.76), so the
cheaper one is quoted and nothing is lost but a confirmation.

### 2026-08-05 — context size: nothing either, once the controls are correct
Calibrated, 5 replicates, only the context cap moving.

| task | 2,500 | 5,000 | 10,000 | 20,000 | 40,000 |
|---|---:|---:|---:|---:|---:|
| rel-avito (86,619 train) | 65.34 ± 0.24 | 65.44 ± 0.28 | 65.54 ± 0.11 | 65.54 ± 0.19 | — |
| rel-trial (11,994 train) | — | — | 69.27 ± 0.86 | 69.35 ± 0.96 | 69.35 ± 0.96 |

**Total range 0.20 on rel-avito and 0.08 on rel-trial — both inside the ±0.6 floor.** The
rel-avito curve is monotone increasing, which is the shape that tempts one to keep going,
but 8× the context buys 0.20. rel-trial's 20,000 and 40,000 cells are *identical* because
both exceed its 11,994 training rows; that is a consistency check on the harness, not two
measurements.

Together with the ensemble sweep above, **the two remaining generic levers are exhausted**:
neither ensemble size nor context size moves these tasks. What has moved numbers on this
branch is task-specific — outcome history on rel-trial (+5.45) and the column budget on
rel-f1 (+12.58) — which is the same lesson as "column budget is a rescue, not a default",
arrived at from the opposite direction.

### 2026-08-05 — ensemble size: nothing, on the two tasks closest to a win
Calibrated, 5 replicates, `max_columns=None`, timed links only, only `n_estimators` moving.

| task | 4 | 8 | 16 |
|---|---:|---:|---:|
| rel-avito | 65.54 ± 0.11 | 65.57 ± 0.13 | *timed out at 3000 s* |
| rel-f1 | 82.48 ± 0.80 | 82.73 ± 0.52 | 82.66 ± 0.41 |

**Every difference is inside the ±0.6 floor**, so 4 stands as the default and the cheaper
setting wins. This was worth checking rather than assuming: every number on this branch was
produced at 4, TabICL's own default is 8, and on rel-event plain features gained six points
going 1 → 8. That gain does not generalise to these tasks.

*One real signal in it:* at `n_estimators=16` on rel-f1, validation switched to `base` 5/5
where 4 and 8 chose `+struct` — the same ensemble-diversity effect that flipped the
categorical-blocks verdict. A wider feature set costs diversity, and a larger ensemble
makes that cost visible in the selection rather than in the score.

*The rel-avito 16 cell is missing, not zero* — it exceeded the 3000 s timeout on 86,619
training rows. Not retried, because 4 → 8 moved 0.03 and there is no reason to expect 16 to
behave differently.

### Table 1 — entity classification, one uniform procedure, val and test
Every row from the same protocol: arm × context size chosen on **validation**, test scored
**once**, 5 replicates, AMP off, timed link tables only, each arm gated by controls on its
own columns. Comparable *with each other*, which the per-task history below is not.

| dataset / task | train | val | test | val AUROC | test AUROC | arm chosen | `max_columns` |
|---|---:|---:|---:|---:|---:|---|---|
| rel-f1 / driver-top3 | 1,353 | 588 | 726 | 87.82 ± 0.06 | **82.48 ± 0.80** | `+struct` 3/5 | None |
| rel-trial / study-outcome | 11,994 | 960 | 825 | 66.77 ± 0.23 | **69.27 ± 0.86** | `+rate` 5/5 | 2 |
| rel-event / user-ignore | 19,239 | 2,013 | 1,958 | 82.16 ± 0.69 | **81.22 ± 3.01** | `+struct` 3/5 | None |
| rel-avito / user-visits | 86,619 | 29,979 | 36,129 | 77.37 ± 0.50 | **65.54 ± 0.11** | `+struct` 5/5 | None |

**Read the val/test columns together — they disagree in both directions and by a lot.**
rel-f1 loses 5.3 from validation to test, rel-avito loses **11.8**, and rel-trial *gains*
2.5. Validation splits of 588–2,013 rows cannot support fine selection, and Rel-LLM's own
Table 1 shows the same pattern (their RDL goes 91.70 val → 81.62 test on rel-event), so
this is a property of the benchmark rather than of our protocol.

*Excluded arms are visible in the runs and matter:* `+history` is excluded everywhere
except rel-avito, `+struct` is excluded on rel-trial, `+rate` on rel-event. Each exclusion
is a control failing on that arm's own columns, and each was a number that would otherwise
have been selectable.

### 2026-08-04 — rel-trial 69.36 CONFIRMED, and rel-event/rel-avito enter at causal values
**rel-trial reproduces: 69.27 ± 0.86, `+rate` chosen 5/5**, against the original 69.36 —
a gap of 0.09, far inside the floor. The A/B repeats the original finding exactly: base
63.82, `+counts` 64.68, `+rate` **69.27**. The earlier scare was mine: folding `n_linked`
into `+history` made control 4 exclude an arm that never contained that column. The `+rate`
arm restores the original composition and passes controls 1–3.

**rel-event 78.11 → 81.22 ± 3.01** and **rel-avito 64.85 → 65.61 ± 0.20**, both with timed
link tables only and all reported arms controlled. Two honest qualifications: rel-event's
spread is 3.01, so its standard error is ~1.35 and the +3.11 is roughly two of them — real
but not comfortable; and rel-avito's +0.76 is barely over the ±0.6 floor. Both come from
`eval_track_record`'s pipeline rather than the one that produced the previous figures, so
they are "our best calibrated number", not like-for-like ablations.

*The column budget did not transfer.* `max_columns=None` is worth +12.58 on rel-f1, and on
rel-avito it changes nothing measurable (65.54 against 65.61 at 2) while on rel-event it is
inside the noise (81.22 against 80.76). One task, one lever — exactly what
"column budget is a rescue, not a default" said, now with a number attached on all four.

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
