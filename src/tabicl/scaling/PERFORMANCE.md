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

## Standing vs published results (updated 2026-08-06)

RelBench, official protocol. Test ROC-AUC ×100. **Seven** of RelBenchV1's twelve entity
classification tasks. Comparison figures are the TabPFN-3 report's Table 14 (arXiv
2605.13986), **complete** except for KumoRFMv1 and RTzero, which the report itself flags as
following a different evaluation protocol that overestimates performance.

An earlier version of this table showed only three comparison methods and reported gaps
against them. That was flattering: RelGT beats us on three tasks and was absent. The full
field is shown here, with our rank in it.

| task | **ours** | rank | best-cfg ‡ | DFS † | RelGNN | RelGT | GraphSAGE | Griffin | RDBLearn | +v2.5 | +v3 | KumoRFMv2 | TabPFN-REL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rel-event / user-repeat ◆ | **77.89** | **3/10** | 78.58 ⁿ²¹ | — | **79.61** | 76.09 | 76.89 | 71.88 | 75.04 | 75.55 | 76.81 | 79.34 | 77.11 |
| rel-trial / study-outcome | **72.26** | **4/10** | 73.56 | 69.12 | 71.24 | 68.61 | 68.60 | 51.00 | 71.58 | 72.90 | 72.89 | 72.03 ˟ | **76.43** |
| rel-f1 / driver-top3 | **81.98** | 6/10 | 84.61 | 76.81 | **85.69** | 83.52 | 75.54 | 82.50 | 79.69 | 77.60 | 82.72 | 82.09 | 79.98 |
| rel-event / user-ignore | **80.98** | 7/10 | 86.77 | 77.95 | **86.18** | 81.57 | 81.62 | 83.27 | 82.52 | 78.65 | 73.70 | 78.86 | 85.38 |
| rel-avito / user-visits | **65.54** | 8/10 | 66.21 | 65.81 | 66.18 | 66.78 | 66.20 | 60.70 | 65.49 | 66.47 | 66.76 | **69.41** ˟ | 66.68 |
| rel-avito / user-clicks ◆ | **65.89** | 8/10 | 68.19 ⁿ²⁰ | — | 68.23 | 68.30 | 65.90 | 45.90 | 69.04 | 65.72 | **69.06** | 67.42 ˟ | 67.09 |
| rel-f1 / driver-dnf ◆ | **69.66** | 9/10 | 69.95 ⁿ⁶ | — | 75.29 | **75.87** | 72.62 | 57.70 | 70.87 | 71.72 | 71.72 | 72.03 | 70.74 |
| **average** ¶ | **73.46** | **6/10** | — | — | **76.06** | 74.39 | 72.48 | 64.71 | 73.46 | 72.66 | 73.38 | 74.45 | 74.77 |

**Median rank 7 of 10.** Ranks: 3, 4, 6, 7, 8, 8, 9. Bold in the comparison columns marks
the best method for that task; **we never hold it.**

¶ **The average row is over these seven tasks only, for every method — so it is internally
comparable but is _not_ the report's Avg AUROC, which covers twelve.** Its rank cell is our
standing among the ten methods' averages, on the same rule as every other row.

**The average ranks us higher than the median rank does, and the difference is the point.**
On mean AUROC we are 6th and on median rank 7th. The 6th is generous even so: RDBLearn is
ahead of us by 0.004 — the two are a tie that the sort had to break. Averaging rewards
*never collapsing* —
Griffin scores 45.90 and 51.00 on two tasks and its average falls 8 points below anyone
else's, while our worst task is still mid-field. So the honest reading is: **we are
consistently mid-field rather than occasionally excellent**, and no averaging convention
turns that into a lead. RelGNN is ahead on both measures and by 2.60 average AUROC.

◆ Added 2026-08-06 — three tasks this pipeline had never been run on. `user-repeat` came
first and placed 3rd; `driver-dnf` and `user-clicks` are the corrective. Running only four
tasks, two of which we do comparatively well on, was itself part of the flattery.

˟ Cells the report imputes from author-supplied results because its own scripts did not
support them; not directly comparable.

† **DFS is measured here, not published.** Featuretools over the same tables with per-row
cutoffs, scored by **the same TabICL, same context, same seeds** — only the feature builder
differs, so it is the one column isolating *our aggregation* from *our model*. We lead 2 of
4 and tie 2, at 5–23× faster feature build. Only run on the original four tasks.

ⁿ **The superscript is the number of configurations the max was taken over**, because a max
over K noisy scores is biased upward and the bias grows with K. Cells without one came from
every variant ever run on that task — far more than 20 — so they are *more* inflated than
the annotated ones, not less. Comparing best-cfg cells across rows compares search effort as
much as headroom.

‡ **An upper bound selected with knowledge of test — never a claim, never bolded.** Included
because the comparison columns are not measured the way our main column is: the report takes
baselines "as provided by the authors… to ensure well-tuned baselines", chose KumoRFMv2's
settings because they "found to slightly outperform" the defaults, and headlines the best of
three RDBLearn variants. Per-row provenance and the reason each entry was rejected by
validation are in the 2026-08-05/06 entries below.

**How to read our position.** The two methods above us most often — RelGNN and RelGT — are
*supervised, per-task trained* graph models; the report classes them separately from
foundation models. Our peer group is the flatten-then-tabular-foundation-model set
(TabPFN-REL, RDBLearn+v3, KumoRFMv2), and **every one of those runs on TabPFN-3 while we run
on TabICL.** Whether our remaining gap to them is featurization or backbone has never been
separated directly. The nearest published evidence — RDBLearn's own three rows, same
features across three backbone generations — puts a generation step at **+0.72 average, sd
3.16**, against our 2.60 deficit, and it is *negative* on the task where our peer gap is
largest. See the 2026-08-06 backbone-ladder entry.

## CORRECTION — our rank against the full field (2026-08-06)

**The comparison table above uses three published methods. The report's Table 14 has ten
comparable ones, and the omitted ones include methods that beat us.** RelGT is ahead on
three of our four original tasks and has never appeared in this file. Reporting a −3.71 gap
"vs best" while the actual field contains four methods above us is a flattering
presentation, and it was not deliberate — it came from copying the paper's *headline*
systems rather than its full table.

Seven tasks now measured, ranked against every method the report does **not** flag as
following a different protocol (KumoRFMv1 and RTzero are excluded on the report's own
advice; ten methods remain, including us):

| task | ours | rank | best in field | methods above us |
|---|---:|---:|---:|---|
| rel-event / user-repeat ◆ | 77.89 | **3 of 10** | 79.61 RelGNN | RelGNN, KumoRFMv2 |
| rel-trial / study-outcome | 72.26 | **4 of 10** | 76.43 TabPFN-REL | TabPFN-REL, RDBLearn+v2.5, RDBLearn+v3 |
| rel-f1 / driver-top3 | 81.98 | 6 of 10 | 85.69 RelGNN | RelGNN, RelGT, RDBLearn+v3, Griffin, KumoRFMv2 |
| rel-event / user-ignore | 80.98 | 7 of 10 | 86.18 RelGNN | RelGNN, TabPFN-REL, Griffin, RDBLearn, GraphSAGE, RelGT |
| rel-avito / user-visits | 65.54 | 8 of 10 | 69.41 KumoRFMv2 | KumoRFMv2, RelGT, RDBLearn+v3, TabPFN-REL, RDBLearn+v2.5, GraphSAGE, RelGNN |
| rel-avito / user-clicks ◆ | 65.89 | 8 of 10 | 69.06 RDBLearn+v3 | RDBLearn+v3, RDBLearn, RelGT, RelGNN, KumoRFMv2, TabPFN-REL, GraphSAGE |
| rel-f1 / driver-dnf ◆ | 69.66 | 9 of 10 | 75.87 RelGT | RelGT, RelGNN, GraphSAGE, KumoRFMv2, RDBLearn+v2.5, RDBLearn+v3, RDBLearn, TabPFN-REL |

◆ new on 2026-08-06. **Median rank 7 of 10.** Ranks: 3, 4, 6, 7, 8, 8, 9.

**This is a materially worse position than this file has been describing**, and the three
new tasks are why it is now visible: running only four tasks, two of which we happen to do
comparatively well on, made the pipeline look stronger than it is. `user-repeat` at 3rd was
the first new result and was flattering; `driver-dnf` at 9th and `user-clicks` at 8th are
the corrective.

**One diagnostic worth keeping from `user-clicks`:** its untuned `base` arm scores **67.18**
against the calibrated **65.89** — validation chose `+struct` (66.08) over `base` and test
punished it by 1.29, which is four places in the field. Calibration made that task worse,
which is the same selection bias documented throughout, showing up where it costs a rank.

## What to expect before and after calibrating (2026-08-06)

The table above is the **after** column: every setting chosen on a validation split. Most
users will not calibrate on the first attempt, so here is what the package gives with no
tuning at all, and what tuning on your own workload is worth.

All figures are test ROC-AUC×100, AMP off. "Out of the box" is the base feature set at
**shipped** settings with no per-task sweep. "Best block" is the best optional feature arm,
still uncalibrated. "Calibrated" is settings chosen on a validation split.

**This table was rebuilt on 2026-08-06 and the reason matters.** Its first version covered
four tasks at `max_columns=2`, and that is no longer the shipped default — it is now **4**,
chosen on validation across four tasks by worst-case regret. A "what you get out of the box"
table describing a budget we no longer ship is worse than no table, because the whole point
of the column is to set an expectation a user can hold us to. Rows are therefore being
re-measured at the current default, and every row says which default it was measured at
rather than blending the two.

**All seven, every row at the shipped defaults, so the "out of the box" column means what it
says:**

| task | out of the box | best block | calibrated | calibration is worth |
|---|---:|---:|---:|---:|
| rel-trial / study-outcome | 69.56 | **72.19** `+rate` | 72.30 | **+2.74** |
| rel-event / user-ignore | 80.22 | **82.09** `+struct` | 81.98 | **+1.76** |
| rel-f1 / driver-dnf | 66.62 | 66.62 (base only) | 68.10 | **+1.48** |
| rel-event / user-repeat | 77.13 | 77.32 `+struct` | 77.89 | +0.76 |
| rel-f1 / driver-top3 | 79.27 | **79.54** `+struct` | 79.90 | +0.63 |
| rel-avito / user-visits | 65.61 | **65.85** `+rate` | 65.50 | −0.11 |
| rel-avito / user-clicks | **67.32** | 66.83 `+counts` | 66.03 | **−1.29** |

**Mean +0.85, negative on two, inside the ±0.6 floor on one.**

**A separate measurement, and do not confuse it with the table.** Both rel-f1 rows were also
run with `--max-columns none`, that dataset's schema-level standing flag, which answers a
different question: *is calibration worth anything once the budget is already right?* Answer:
**no — −0.06 on driver-top3 and −0.29 on driver-dnf.** The gap between the two runs prices
the shipped default on that schema: **untuned, `none` beats the shipped 4 by 2.92 on
driver-top3 and 2.57 on driver-dnf.** So on rel-f1 the shipped budget costs more than
calibration recovers, and calibration's +0.63 and +1.48 above are mostly it clawing back
ground the default gave away. I first wrote these two rows into the table labelled
`max_columns=4` when they had been run with `none`; that was wrong, and both figures now
appear where they belong.

**THE DECISIVE RESULT, and it splits the prediction rather than settling it.** driver-top3
carried the +8.48 that dominated the old table, and I predicted that fixing the budget would
collapse it. **On driver-top3 it collapses completely: +8.48 → −0.06.** Almost the entire
apparent value of calibration on that task was one global default being wrong for that
schema. That is the mechanism, confirmed on the task it was proposed for.

**But it does not generalise.** rel-trial gains **+2.74** and rel-event +1.76 and +0.76,
where the budget was never the issue — on those a wider budget widens the gap between the
default arm and the best arm, and selection is what captures that. So the honest statement
is neither "calibration is budget repair" nor a single average: **where a global default is
wrong for your schema, calibration repairs it and is worth a great deal; where the defaults
already fit, it ranges from −0.3 to +2.7 and the sign varies by schema.**

**The clearest pattern is not the average — it is which datasets gain.** rel-trial, rel-event
and rel-f1 gain; **both rel-avito tasks lose.** That split is not about the amount of tuning
available, it is about whether this schema gives the pipeline anything to tune. rel-trial's
best block is `+rate` at +2.63 over base and rel-event's is `+struct` at +1.87; on rel-avito
the best block is worth +0.24 on one task and *nothing* on the other, where the untuned
`base` at **67.32** beats every block and beats the calibrated 66.03.

**rel-avito is the case worth telling a user about**, because it is the one where tuning
actively hurts, twice, and the mechanism is visible: with no block worth choosing, selection
is choosing between near-identical options on a small validation split, so it is fitting
noise. `user-clicks` loses **1.29** that way — four places in the published field — and the
figure reproduced almost exactly across two independent rounds (67.18 → 65.89 at the old
default, 67.32 → 66.03 at the new one), so it is a property of the task rather than a bad
draw.

**One thing the table shows that the calibrated protocol missed.** On rel-avito/user-visits
the best block is `+rate` at **65.85** — better than base 65.61 and better than the
calibrated 65.50. The selection rule had that arm available and did not take it. That is the
same selection bias documented throughout, and it is worth noting that `+rate` here is the
User → Ad → User similarity feature, eligible on this task but excluded on user-clicks by
the broken control fixed above.

**The legacy figures, kept because they are what the earlier conclusions were drawn from.**
At `max_columns=2`: driver-top3 73.50 → 81.98 (**+8.48**), rel-trial 69.56 → 72.26 (+2.70),
user-ignore 80.80 → 80.98 (+0.18), user-visits 65.81 → 65.54 (**−0.27**), user-clicks 67.18
→ 65.89 (**−1.29**), driver-dnf 68.99 → 69.66 (+0.67). **Mean over the four originals:
−0.18.** Almost all of driver-top3's +8.48 was the column budget being wrong for that schema
by 12.58 on validation — which is a bad global default repaired per task, not per-task
selection earning its keep, and is exactly why the default was changed.

**A user-facing summary that survives both tables.** Calibration is worth somewhere between
−1.3 and +8.5 depending on your schema, it is **negative on two of seven tasks**, and the
single biggest determinant is whether the shipped column budget suits your child tables. If
you have a wide child table like rel-f1's `results`, tune; if your tables are narrow, the
defaults are close to what tuning would find.

**The lever that matters is schema-dependent, and that is the useful thing to tell someone
before they start:**

* **rel-f1 — calibration dominates (+8.48).** Almost all of it is the column budget: the
  shipped `max_columns` is wrong for this schema by 12.58 on validation. Opting into
  feature blocks adds only +0.55.
* **rel-trial — the feature blocks dominate (+2.64).** The shared-key track record carries
  it; calibration on top adds +0.06.
* **rel-event — neither helps (+0.18).** The defaults are already at this schema's ceiling,
  and its optional blocks actively hurt (`+struct` −0.41, `+counts` −0.56).
* **rel-avito — calibration is negative (−0.27).** Out of the box beats our own calibrated
  number. Inside the ±0.6 floor, so read it as a tie rather than a loss — but the arm and
  context machinery buys nothing on this schema.

So the honest expectation to set is a **range, not an uplift**: somewhere between −0.3 and
+8.5, depending mostly on whether the shipped column budget suits the schema. A user whose
tables are narrow will see little; one with a wide child table like rel-f1's `results` will
see a lot.

**Caveat on "out of the box", and it is a real one.** The legacy figures use
`eval_track_record`'s old defaults — `max_columns=2`, windows on, three child tables. The
library's `Table` defaults are **not the same**: `max_columns=None` and `windows=()`. A user
calling `flatten_relational` directly therefore gets no windows and no column budget, which
on rel-f1 should land *closer* to 81.98 than the 73.50 above. The three places that define
a default — the library, the runner, and `STATUS.md`, which calls the budget "a rescue, not
a default" — currently disagree. `eval_defaults.py` picks each axis on validation across all
four tasks so they can be reconciled with a measurement rather than by inheritance.

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

**Re-run at 5 seeds, and validation still cannot distinguish them:**

| child set | val | test |
|---|---:|---:|
| first 3, dict order | 68.65 ± 0.65 | 72.32 ± 0.94 |
| best 3 of 10, ranked on validation | 68.38 ± 0.69 | **73.37 ± 0.30** |

Test says **+1.05** with the spread tightening threefold. Validation says **−0.27** against a
combined SE near 0.42 — indistinguishable, and leaning the wrong way. **So the gain is not
claimed**, and this becomes the sixth effect in this log that is real on test and not
selectable.

**But the comparison has a defect the scores cannot fix.** Dict order is not a
specification: it selects a different set of tables on different machines, so the 72.32
baseline does not name a reproducible configuration. Among two options validation cannot
separate, there is a case for preferring the one that is well-defined — and `--top-children`
uses only validation data to rank, so choosing it costs no test information.

That is a decision about *specification*, not a measured improvement, and it is left open
rather than taken unilaterally: adopting it would move rel-trial's headline to 73.37 on an
argument rather than on a measurement. `--children-order name` is the other defensible
specification, and it is arbitrary but neutral. The one option that should not stand is the
current default.

### 2026-08-06 — rel-trial categories settled at 12 replicates: real, and not selectable

Three children (the standing configuration), paired per seed, twelve replicates:

| | none | `--categories 4` | paired Δ |
|---|---:|---:|---|
| test | 72.43 ± 0.73 | **73.21 ± 0.57** | **+0.78** (SE 0.186, **4.2 SE**, 11/12) |
| val | 68.49 | 68.36 | **−0.13** (SE 0.101, 1.3 SE, **3/12**) |

**Test gain is solid. Validation mildly disprefers it, consistently** — nine of twelve seeds
negative. So this joins the unselectable list rather than the table.

**Two readings of mine that twelve replicates killed**, both from five-seed data:

* "Validation prefers categories (+0.09)" — the sign was split 3-of-5 even then, and at
  twelve it is −0.13 with 3/12 positive.
* "68.50 is the highest validation score of any configuration tested" — noise. The margin
  was 0.09.

The 3-replicate ten-children result (Δval +0.43, 3/3) should be read the same way: three
replicates on a task whose paired val SE is 0.10 at twelve.

**The baseline reproduces**: 72.43 ± 0.73 against a standing 72.26, so the configuration is
sound and only the verdict changed.

---

### 2026-08-06 — fitting on train+val: large, task-dependent, and undecidable from validation

We fit the final model on **train only**. RelBench holds out test alone, so training on
train+val is permitted, and refitting on everything after selecting on a held-out split is
standard. Paired per seed, each task at its own standing configuration:

| task | train only | train+val | paired Δ | SE | positive |
|---|---:|---:|---:|---:|:---:|
| rel-f1 | 82.13 | **84.61** | **+2.48** | 0.58 | 7/8 |
| rel-trial | 72.30 | **73.56** | **+1.26** | 0.50 | 6/8 |
| rel-avito | 65.67 | 65.78 | +0.11 | 0.12 | 3/5 |
| **rel-event** | 81.58 | 74.29 | **−7.29** | 1.56 | **0/5** |

**The mechanism for the positive side**: it helps when the *pool* binds the context, not
when the *cap* does. rel-f1's context is 1,353 — all of train — so 588 extra rows go
straight in. rel-avito's cap is 10,000 against 86,619 available, so a bigger pool changes
nothing. This predicted rel-trial's moderate gain before it was measured.

**rel-event's regression is likely the arm**: it selects `+struct`/`+counts`, whose
structural counts *grow with time*, so validation rows sit at systematically higher values
than training rows and poison the context. rel-f1 selects `base`, which has no history
features at all, and gains most.

**It cannot be adopted, and the reason is the same one as everything else this week.** The
ordinary validation column is *identical* in both arms — the selection is unchanged, only
the final fit differs — so the usual instrument is structurally blind.

`--decide-fit-pool` was built to fix that without touching test: split validation by time,
let the earlier 60% join the pool, judge on the later 40%. **It is anti-correlated with the
truth:**

| task | better option | procedure chose | cost |
|---|---|---|---|
| rel-event | train only | **train+val, 4/5** | −4.90 |
| rel-f1 | train+val | **train only, 8/8** | forfeits +2.48 |

Adding early-validation rows helps predict late-validation rows — they are adjacent in time
— and that does not transfer to test. **A third instrument, broken the same way as the
first two.**

So the options are: per task by test score (test-selection, refused); uniformly (+2.48
+1.26 +0.11 −7.29 = **net −2.52**, knowingly worse); or by this procedure (wrong on both
tasks that matter). **Train+val fitting is the ninth real-on-test, unselectable effect** —
and the first whose effect is strongly negative somewhere. The standing table is unchanged.

### 2026-08-06 — we hold ourselves to a stricter standard than the column we compare against

From the TabPFN-3 report's own methods section (arXiv 2605.13986, §3.4), which is the
source of every comparison figure in the table at the top of this file:

* *"we generally report baseline results as provided by the authors of the methods to
  ensure well-tuned baselines"* — so RelGNN's 86.18 is the authors' own tuned figure.
* *"For KumoRFMv2 we ... use four estimators and a context size of 10000 (the respective
  maxima for each), **which we found to slightly outperform** the script defaults"* —
  settings chosen by observed performance.
* *"We compare **three different versions** of RDBLearn"*, and the headline figure is the
  best of them (RDBLearn + v3).

And Figure 21 splits the field into **foundation models** (TabPFN-REL, KumoRFM, Griffin,
RTzero) and **supervised, per-task tuned** (RelGNN, RelGT, GraphSAGE, RDBLearn). **RelGNN,
the number this project has been chasing, is in the second class.** Our own class is
TabPFN-REL, against which the standing table is +2.00 on rel-f1, −1.14 on rel-avito, −4.17
on rel-trial and −4.40 on rel-event.

Meanwhile our protocol chooses every setting on a validation split, touches test once, and
has this week **declined nine measured effects** because validation did not endorse them.
That is materially stricter than author-reported-best-tuned or best-of-three-variants.

**This does not license test-selection**, and nothing in the table changes on the strength
of it. But it does mean the calibrated column is not like-for-like, and it is conservative
in the *unfavourable* direction. The paper's own practice — reporting three RDBLearn
variants — suggests the honest alternative is two columns rather than a looser one:

| task | calibrated (validation-selected) | best measured configuration |
|---|---:|---:|
| rel-f1 | 81.98 | 84.61 — train+val fitting |
| rel-event | 80.98 | 86.77 — recent 1,000-row context |
| rel-avito | 65.54 | 66.21 — top-1 key |
| rel-trial | 72.26 | 73.56 — train+val fitting |

The second column is **an upper bound selected with knowledge of test** and must be
labelled as such wherever it appears. It is not a claim of superiority and must never be
the headline — but hiding it while comparing against best-of-variants figures is its own
kind of misreporting. **Left as an open decision for the maintainer**; the headline table
is unchanged.

*Also relevant to reading that comparison: the report marks KumoRFMv1 and RTzero as
"likely following a different evaluation protocol than the one outlined in RelBench, which
overestimates model performance". Protocol rigour is a known problem on this benchmark,
which is the strongest argument for keeping the calibrated column as the headline and
publishing the unselectable finding rather than loosening what we report.*

### 2026-08-06 — the context cap was not costing anything

`--context` defaults to 10,000 and the calibrated grid is `{cap//4, cap//2, cap}`, so the
sweep had never looked above it: 10,000 of rel-avito's 86,619 training rows, 10,000 of
rel-event's 19,239. An unexamined default of exactly the kind that has paid off repeatedly
this week — and the one axis where the diagnosis predicted validation would *cooperate*,
since it demonstrably rewards context size.

| task | context | val | test |
|---|---:|---:|---:|
| rel-avito | 10,000 | 77.34 | 65.67 |
| rel-avito | 20,000 | 77.72 | 65.70 |
| rel-avito | 40,000 | **77.83** | 65.74 |
| rel-event | 10,000 | 87.12 | **81.58** |
| rel-event | 19,000 | **87.53** | 80.68 |
| rel-trial | 10,000 | 68.52 | 72.30 |
| rel-trial | 12,000 (control) | 68.55 | 72.56 |

**+0.07 on rel-avito for four times the context, −0.90 on rel-event.** The cap was not a
limitation, and this closes the last open lead.

Two things worth keeping from it. Validation rose monotonically with context on every task
— 77.34 → 77.72 → 77.83 on rel-avito, 87.12 → 87.53 on rel-event — which confirms the
"validation rewards context size" diagnosis a third and fourth time, independently of the
recency work that produced it. And rel-event shows the same inversion yet again: validation
up, test down.

This is the mirror image of the nine unselectable effects: **selectable and worthless**.
Validation and test agree, and there is nothing to collect.

### 2026-08-06 — what the defaults should be: `max_columns=4`, and both current values are wrong

The defaults were never chosen, and they disagree with each other:

| setting | library `Table` | `eval_track_record` | `STATUS.md` |
|---|---|---|---|
| `max_columns` | `None` | `2` | *"a rescue, not a default"* |
| `windows` | `()` — none at all | per-dataset | *"the strongest signal a history carries"* |

So a user calling `flatten_relational` gets no windows and no budget, while every published
number here was measured with windows and a budget of 2.

A default is chosen without seeing the user's data, so the criterion is robustness across
schemas. Scored on **validation**, all four tasks, 3 seeds, test never read.

**`max_columns`:**

| task | `None` | `2` | `4` | note |
|---|---:|---:|---:|---|
| rel-f1 | **87.18** | 68.05 | 85.05 | 428 columns at `None` |
| rel-trial | 65.38 | 65.38 | 65.38 | budget never binds — under 2 columns per child |
| rel-event | 81.80 | 87.54 | **87.92** | **2,000 columns and 349 s** at `None` |
| rel-avito | **69.32** | 69.28 | 69.23 | |

Mean rank picks `None` (1.50 against 2.25 for both others), **but that is the wrong
criterion.** A default should bound the damage when it is wrong, so the statistic is
worst-case regret against the best value for each task:

| candidate | worst-case regret |
|---|---:|
| `None` | 6.12 (rel-event) |
| `2` | **19.13** (rel-f1) |
| **`4`** | **2.13** (rel-f1) |

**`max_columns=4`.** Never more than 2.13 off the best on any task, against 19.13 for the
value the runner ships and 6.12 for the value the library ships. Both current defaults are
wrong, in opposite directions, and `4` was in neither place.

**`windows`:** `single` (a fixed 30 days) ranks 1.75, `task` (hand-picked per dataset) 2.00,
`none` 2.25 — all within one rank, and the ordering is inconsistent across tasks (`single`
places 3rd, 1st, 1st, 2nd). Two things follow. The hand-picked per-task windows are *not*
better than a uniform 30 days, including on rel-trial where 30 days beats 365/1095 — so
those values were never validated either. And windows nearly triple the feature count
(44 → 122 on rel-f1) and roughly double build time (27 s → 55 s on rel-avito) for a
difference inside noise, which makes the cost argument stronger than the accuracy one.

### 2026-08-06 — ensembling over configurations: no accuracy effect

The idea was to route around the constraint rather than fix it. Nine effects are real on
test and the validation split cannot rank them reliably; three instruments built to repair
that all failed. Ensembling never has to choose — average the top N configurations by
validation score instead of taking the argmax.

Calibrated, 5 replicates, `--max-columns 4`:

| task | argmax | ensemble N=5 | Δ mean | sd |
|---|---:|---:|---:|---|
| rel-event | 81.72 | 81.41 | **−0.31** | 1.06 → **1.45** |
| rel-f1 | 82.18 | 82.19 | +0.01 | 1.04 → **0.41** |
| rel-trial | 72.32 | 72.33 | +0.01 | 1.08 → 1.04 |
| rel-avito | 65.50 | 65.61 | +0.11 | 0.21 → **0.12** |

*(rel-event at N=3: 81.45, sd 1.18 — monotone between the two.)*

**Mean effect across the four tasks: −0.045. Nil.** The variance effect is inconsistent —
down 2.5× on rel-f1 and 1.8× on rel-avito, *up* on rel-event, flat on rel-trial — and it
tracks how homogeneous each task's candidate pool is rather than anything about the method.
rel-f1's six candidates are near duplicates, so averaging removes the seed-to-seed noise of
*which* one gets picked; rel-event's nine include genuinely weaker configurations that N=5
drags into the average.

Cheaper reproducibility on some tasks is worth something, but it is not what this was
proposed for and is not an accuracy result.

**In hindsight the null is unsurprising and the test was badly aimed.** The candidates the
sweep offers are arms × context sizes, which sit close together; averaging configurations
that already agree cannot produce information. The 3.9-point gap that motivated this was
between context *orders*, which were not in the pool.

So the question is narrower than it was posed: does ensembling help when the pool is
**heterogeneous** and the argmax is **wrong**? rel-event with `--context-orders` is the one
case in this project where both hold — the argmax there picks `recent@10000` (test 79.75)
over `random@10000` (~82). Ensembling can only ever help by diluting a bad argmax, and the
test above never presented it with one.

### 2026-08-06 — preserving missingness is worse than zero-filling it

The aggregation deliberately keeps "no history" distinct from "zero" — *"a missing mean is
not 0"*, and for recency *"0 would assert the opposite of what is true"* — and then every
runner ends with `nan_to_num(nan=0.0)`. TabICL declares `allow_nan` and preprocesses with
`nanmean`/`nanstd`, so the zero-fill looked inherited rather than chosen.

**It is chosen, and it is right.** Paired by seed, same features, only the final cast
differs:

| task | NaN share once preserved | gap |
|---|---:|---:|
| rel-f1 | 4.8% | −0.06 (SE 0.17) — null |
| **rel-event** | **54.9%** | **−1.03** (SE 0.48, 1/5) — clears, negative |
| rel-avito | 42.0% | +0.12 (SE 0.25) — null |
| rel-trial | ~44% | crashed; see below |

**No benefit anywhere, and one real cost.** The zero-fill is chosen rather than inherited,
and the docstrings describe an intent the pipeline overrides — correctly.

*A mechanism I asserted and then had to withdraw.* On rel-event's result alone I wrote that
preserving NaN "hurts where there is a lot of it", reasoning that `nanmean`/`nanstd` have
too little to work with at 55% sparsity. rel-avito is 42% missing and is unaffected. Two
tasks with comparable sparsity and opposite outcomes, so **NaN share is not the
explanation** and the honest summary is: negative on rel-event, null on the other two that
ran, cause unknown.

*A correction to how this was motivated:* "44–47% NaN" was quoted as if typical. It is
rel-trial's figure. rel-f1 is 4.8%, so there was nothing there to change and its null says
nothing either way.

**A real TabICL bug, found on the way and not chased.** With NaN preserved, rel-trial dies
inside `predict_proba`:

```
preprocessing.py:1172  filtered_mask = feature_mask[self.unique_filter_.features_to_keep_]
IndexError: size of axis is 134 but size of corresponding boolean axis is 103
```

`classifier.py:804` builds `feature_mask` over the *input* feature space, and it is then
indexed against a filter fitted in a *reduced* one. The path is only reachable when some
test column is entirely NaN, which sets `feature_mask` non-None — and zero-filling makes
that impossible, so it has never been exercised despite `allow_nan = True` advertising that
it should work. A 60-row synthetic case with a constant train column and an all-NaN test
column does **not** reproduce it, so the trigger is more specific than that. Left documented
rather than fixed: it is core inference code, a blind fix there is riskier than the bug, and
the measurement above says we do not want NaN input regardless.

### 2026-08-06 — the selection problem is BIAS, not noise, and that closes ensembling

The strong version of the ensembling test: rel-event with `--context-orders` in the
candidate pool, so the ranking spans genuinely different configurations and the argmax is
known to pick badly there.

| | argmax | ensemble N=5 |
|---|---:|---:|
| test | 80.30 ± 1.24 | 80.84 ± 1.52 |

**+0.54 at roughly 0.5 SE on 3 replicates — null**, like the weak version before it. But
the *reason* is visible in what it averaged. The top five by validation, every replicate:

```
+struct@10000/recent, +counts@5000/recent-half, +counts@10000/recent,
+struct@5000/recent-half, +struct@10000/recent-half
```

Almost entirely recency configurations, and the argmax chose `recent` 3/3. **The whole top
of the ranking is wrong, not just its first element.**

**That is the distinction this project has been missing.** Ensembling defends against a
*noisy* ranking — one whose ordering wobbles around the truth. rel-event's validation is
not noisy, it is **biased**: it systematically prefers recency and larger contexts, because
it observes a period nearer to train than test is. Averaging the top of a biased ranking
averages several versions of the same mistake.

Noise can be averaged away. Bias cannot. This explains, in one sentence, why:

* three selection instruments failed, the last anti-correlated with the truth;
* ensembling was never going to work, in either version;
* nine effects are real on test and unclaimable;
* and the failures were *consistent in direction* rather than scattered, which noise would
  not produce.

**Ensembling is closed.** So is the search for a better instrument built from the same
validation split — anything derived from a biased signal inherits the bias. An instrument
that worked would need information the split does not contain: either a later validation
period, or knowledge of how the test distribution differs.

### 2026-08-06 — test-time compute cannot move AUC, and the code says so before the GPU does

`think_predict_proba` is item 4 of the package's four scaling techniques, listed in
`STATUS.md` as "Working, modest", and it appeared **nowhere** in this file — never
benchmarked on RelBench.

**The prediction, recorded before running**, from reading `_ttc.py`:

* `_mean_proba` varies only `random_state` across refits — model-seed ensembling at a fixed
  context, which is what `n_estimators` does and which this log records as measured at
  nothing on every task.
* The head is a `LogisticRegression` over the backbone's own 2-column probability matrix,
  so the blend is `(1−w)·p + w·σ(αp + β)`, **monotone in p when α > 0. ROC-AUC is invariant
  under monotone transforms of the score**, so the head can leave AUC alone or hurt it,
  never help.
* Its blend weight is chosen on **log-loss**, which is not the metric here.

**Measured, 3 seeds, paired:**

| task | gap | blend weights per seed | permutation effect |
|---|---:|---|---|
| rel-trial | +0.12 (SE 0.23, 2/3) | 0.00, **0.90**, 0.30 | `val_base ≈ val_thought` to 4 dp |
| rel-event | −0.55 (SE 0.70, 1/3) | 0.15, 0.15, 0.00 | `val_base ≈ val_thought` |

**The diagnostic confirms the mechanism, not merely the null.** The head *engaged* — up to
`w = 0.90` — because the guard genuinely wanted it on log-loss, and AUC did not move
anyway. A null with `w = 0` everywhere would have proven nothing. And `val_base ≈
val_thought` on every seed says permutation averaging changed even the log-loss by nothing,
which is `n_estimators` measuring null again by another route.

So: **3× the fit cost, no AUC.** Closed. `STATUS.md`'s "Working, modest" is accurate about
*calibration* and misleading for this benchmark, which scores ROC-AUC — a metric this
feature is structurally unable to improve.

*Method note:* this null cost ten minutes of reading and two tasks instead of a four-task
sweep, because the prediction was derived from the implementation first. After a day of
expensive nulls that is the cheaper order of operations, and it is the same habit —
read what the code actually does — that produced nine bug findings.

### 2026-08-06 — comparability of the headline numbers, verified rather than assumed

The TabPFN-3 report describes its protocol as *"truncating each database at the
pre-specified test timestamp before constructing the featurization and context for all test
entities"* — a **single global cutoff**. This pipeline uses **per-row** cutoffs, each entity
seeing child rows up to its own timestamp. If the shipped database contained rows between
the test timestamp and a late test row's own time, our features would use data the published
methods exclude, and no number in the table above would be comparable.

Checked, and it does not:

| task | distinct test timestamps | test span | child rows at/after `test_timestamp` |
|---|---:|---|---|
| rel-f1 | 30 | 2010-03-02 … 2013-03-16 | **0** of 52,520 |
| rel-trial | 1 | single day | 3,028 of 4,671,285 (0.06%) |
| rel-event | 1 | single day | **0** of 10,904,791 |
| rel-avito | 1 | single day | 6 of 7,509,452 |

Three tasks have one test timestamp, equal to `test_timestamp`, so per-row and global
cutoffs coincide exactly. rel-f1's test rows run three years past its cutoff and **no child
row exists after it** — RelBench ships the database pre-truncated, so there is nothing there
to use. The residuals on rel-trial and rel-avito are rows sitting exactly *at* the boundary,
and the filter is a strict `<`, so they are excluded regardless.

**Our per-row cutoff is therefore no more permissive than the published protocol.** This is
recorded because every comparison in this file depends on it and none of them had checked
it — the kind of assumption that is invisible until it is wrong.

### 2026-08-06 — rel-event's 11.68-point spread is label-specific, and reads like selection luck

Two published methods share the **same backbone** (TabPFN-3) and differ enormously on our
worst task. But rel-event carries *two* tasks in RelBenchV1, and the same pair sits
differently on each (report Table 14):

| method — both TabPFN-3 | rel-event / **user-repeat** | rel-event / **user-ignore** |
|---|---:|---:|
| TabPFN-REL | 77.11 | **85.38** |
| RDBLearn + v3 | 76.81 | **73.70** |
| difference | **0.30** | **11.68** |

Same database, same backbone, same two featurizers: **0.30 apart on one label and 11.68 on
the other.** A genuine featurization advantage would show on both. This one does not, so it
is not a property of how either method builds features from this schema.

**And `user-ignore` is the task where this project independently measured validation to
anti-predict test — three separate times**: recency (+7.50 on test, rejected 3/3), calendar
(+3.00 on test, rejected at 4.7 SE paired), and the earlier categorical result (validation
81.63 vs 81.40 while test said −3.52). On a task that hostile to model selection,
RDBLearn+v3's 73.70 is plausibly *its selection instrument failing*, not its features being
worse — the same failure mode documented at length above, showing up in someone else's
numbers.

**What this changes about our own position.** The −5.20 gap on rel-event is partly a
statement about that task's leaderboard rather than about our features: the spread between
two same-backbone methods there is more than twice our gap to the best of them. Our 80.98
sits between the two, produced by a protocol that declines anything validation does not
endorse. It is a *conservative* estimate on a task where the published spread suggests
selection luck is worth ~12 points.

*This is an interpretation of published numbers, not a new measurement, and it changes
nothing in the table.* It is recorded because "we are 5.20 behind on rel-event" and "this
task's numbers move 11.68 on selection alone" are very different readings of the same gap,
and the second is better supported.

### 2026-08-06 — rel-event/user-repeat: a new task, run to try to falsify our own claim

The previous entry argued that rel-event/user-ignore's 11.68-point published spread between
two same-backbone methods looks like selection luck rather than a featurization gap, partly
because those methods sit only 0.30 apart on **user-repeat**. That argument has an obvious
failure mode: if our own features are simply weak on this schema, the −5.20 on user-ignore
is a feature deficit and the interpretation is wrong.

user-repeat is the control, and it had never been run here. **The falsification condition
was set before the run:** land competitively (~76–79) and the claim holds; land well below
the field and withdraw it.

Calibrated, 8 replicates, `--timed-links-only`, current defaults, no tuning for this task:

| method | user-repeat | vs ours |
|---|---:|---:|
| RelGNN | **79.61** | −1.72 |
| **ours** | **77.89 ± 1.47** | — |
| TabPFN-REL | 77.11 | +0.78 |
| GraphSAGE | 76.89 | +1.00 |
| RDBLearn + v3 | 76.81 | +1.08 |
| RelGT | 76.09 | +1.80 |

**Second of six.** The claim survives: the same features, same pipeline and same schema
place second on one label and −5.20 on the other, so rel-event featurization is not the
problem — the problem is specific to `user-ignore`, which is exactly the task where
validation was measured anti-predicting test three separate times.

**Margins, stated properly.** 246 test rows give SE 0.52. +0.78 over TabPFN-REL is 1.5 SE —
a tie-to-slight-lead, and *not* a second win to put beside rel-f1. The −1.72 to RelGNN is
3.3 SE and real. The base arm before any calibration is 77.13, already level with
TabPFN-REL, with calibration worth +0.76.

*Fixed on the way:* `REFERENCE` was keyed by dataset alone, so this run printed
user-ignore's comparison figures (85.38, 86.18) beside a 77.89. A wrong reference is worse
than none — it invites reading a score against a different label's leaderboard. Now keyed
by `(dataset, task)`.

## Session close, 2026-08-06 — the table is unchanged, and that is the finding

**rel-f1 81.98 · rel-event 80.98 · rel-avito 65.54 · rel-trial 72.26.** Nothing entered.

**Every gate result reproduced on test. None of the large ones survived selection.**

| candidate | task | gate | Δtest (paired) | Δval (paired) | outcome |
|---|---|---:|---:|---:|---|
| calendar | rel-event | +2.96 | **+3.00** (4.9 SE) | **−1.91** (4.7 SE) | rejected |
| categories | rel-trial | +0.90 | **+0.78** (4.2 SE) | −0.13 (1.3 SE) | rejected |
| recency | rel-event | +7.50 | — | validation picks large contexts | rejected |
| timing | rel-f1 | +0.62 | +0.02 at the correct budget | — | null |
| narrow | rel-trial | +0.99 | +0.28 | −0.45 | rejected |

**Eight effects are now real-on-test and unselectable.** The mechanism is specific:
**validation rewards context size**, and on rel-event it accepts `recent` only at 10,000
rows — the diluted version worth +0.31 — while the +7.50 lives at 1,000. More context helps
on a *nearby* period; a small recent context helps on a *distant* one; validation observes
only the first. `--gap-validation` was built to fix exactly this and **made things worse**
(78.71 against 80.25), because its pseudo-split scores 92.76 and selects on a smaller pool.

**What actually moved the work forward was auditing what the code does not do.** Eight
defects, six of which printed output shaped like a careful result rather than an error: a
column budget deleting a feature block; the target passed in as a feature (AUC 100.00 in
*both* arms, so the gap read as a clean +0.00); an all-NaN block the model silently drops;
a flag keyed on the wrong dtype so it never fired; a constant column read as a rate; and a
`tail -60` discarding four variants per task. Plus two that were worse in kind: `--children
3` selecting **different tables on different machines**, and promotions not inheriting the
per-task configuration they were being compared against.

**Two of my own hypotheses were refuted by measurement**: that heavy-tailed count features
want `quantile` normalization (−3.07 on rel-trial, negative on all four), and that a
gap-matched validation split would fix the selection instrument.

### 2026-08-05 — gap-matched validation: my own fix, tested and refuted

The diagnosis was that RelBench's validation split sits nearer to train than test does
(7 days against 15 on rel-event), so a setting chosen on it is tuned for a shorter horizon
than the one it is scored at. `--gap-validation` carved a pseudo-validation split out of
train whose distance from its fitting pool matched the train→test gap.

**It made things worse.** rel-event, calibrated, `--timed-links-only`, 3 replicates:

| instrument | order chosen | pseudo/val score | test |
|---|---|---:|---:|
| ordinary validation | `recent` 3/3 | 88.02 | 80.25 ± 0.45 |
| **gap-matched** | `random` 2/3 | 92.76 | **78.71 ± 1.78** |
| *(no context orders offered)* | — | 87.13 | 82.70 ± 1.08 |

The pseudo-validation split scores **92.76**, far above real validation (~88) and test
(~80): the last 2,013 training rows are simply easier to predict than either. And selection
ran on an 11,522-row pool while the final fit uses 19,239, so the context size it picks is
tuned for a smaller fitting set. **The instrument's own bias is larger than the one it was
built to remove.**

**The sharper diagnosis, which the per-configuration scores make visible:**

```
+counts/recent       context=5000   val=84.49
+counts/random       context=10000  val=87.54
+counts/recent       context=10000  val=88.00   ← chosen, test 79.75
```

Validation **rewards context size**. It will take `recent` only at 10,000 rows — half the
training period, the diluted version worth +0.31 — and scores small recent contexts *low*.
The +7.50 lives at 1,000 rows. So the failure is not merely that validation is nearer to
train; it is that **more context helps on a nearby period while a small recent context helps
on a distant one**, and validation can only observe the first.

That is a property of the benchmark's split design, not something a fold arrangement fixes.
**The selection thread is closed, this time with the alternative built and measured rather
than assumed** — which is the distinction I failed to draw when I closed it the first time.

### 2026-08-05 — the standing configuration is per-task, and promotions must carry it

rel-f1's timing promotion scored **73.21** against a standing **81.98**. Not noise, not a
regression: rel-f1 requires `--max-columns none` — the +12.58 finding from earlier in this
project, listed explicitly in Table 1 — and the promotion ran at the default `2`.
`--timed-links-only` is a no-op there (5 timed link tables, 0 untimed), so the column budget
is the entire difference.

Three configuration mistakes in one round, all the same shape — assuming a promotion
inherits the settings that produced the number it is trying to beat:

1. rel-event's first promotion omitted `--timed-links-only`, and rel-event is the **only**
   task where that matters (2 untimed `user_friends` tables against 0 elsewhere).
2. I then applied that caveat to rel-avito and rel-trial, where all link tables are
   timestamped and the flag does nothing.
3. rel-f1's promotion used the default column budget instead of the uncapped one its
   headline number depends on.

The internal comparisons survive all three — both arms of each pair shared the wrong
setting, so the *difference* is still one variable. Only the absolute numbers are
incomparable to the table. But a promotion whose absolute number cannot be compared to the
thing it is promoting against is half a measurement.

**What this needs is a per-task configuration record**, not more care: `REFERENCE` in
`eval_track_record` already names each task's standing score, and the flags that produced
it belong beside it.

### 2026-08-05 — rel-trial categories: the one thing today that validation and test agree on

Calibrated at **ten** child tables, paired per seed (seed *i* draws the same rows and the
same model randomness in both arms):

| seed | none val | `--categories 4` val | Δval | none test | categories test | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| 0 ¹ | 67.81 | 68.42 | +0.61 | 73.42 | 74.09 | +0.67 |
| 1 | 67.69 | 68.29 | +0.60 | 73.39 | 73.81 | +0.42 |
| 2 | 68.35 | 68.43 | +0.08 | 71.08 | 71.21 | +0.13 |

**Δval +0.43 (SE 0.17, 2.5 SE, 3/3). Δtest +0.41 (SE 0.16, 2.6 SE, 3/3).**

¹ Seed 0 reconstructed from the reported mean and verified against the printed range — a
`tail -20` in the run script cut the first line, for the third time today.

**Validation prefers it and test agrees.** After a day of effects that are real on test and
invisible to the selection rule, this is the one that is selectable. It is small — +0.41,
below the ±0.6 rule of thumb — but that rule was calibrated for unpaired comparisons; at
2.6 SE with 3/3 agreement it is a real effect, and the selection rule adopts it without
being told to.

`--budget-categoricals` did **not** survive the same test: +0.28 on test, −0.45 on
validation, against a gate result of +0.99. Its gate gain came from removing 38 stray
`nunique` columns, and the calibrated sweep can already compensate by choosing a different
arm or context size.

**Why this does not move the headline by itself.** The standing rel-trial number uses
*three* children, and validation ranks three-children-none (68.65) above
ten-children-with-categories (68.38). That comparison is unpaired — five seeds against
three — so it is not yet trustworthy in either direction, and the 2×2 is being closed at
matched seeds. Until then the honest statement is conditional: *at ten children, turn
categories on.*

### 2026-08-05 — calendar on rel-event: +3.00 on test, and the validation verdict is UNDECIDED

Calibrated, `--timed-links-only`, 3 replicates each:

| flags | val | test |
|---|---:|---:|
| none | **87.13 ± 0.43** | 82.70 ± 1.08 |
| `--calendar` | 85.21 ± 1.12 | **85.70 ± 0.21** |
| `--calendar --calendar-trend` | 85.25 ± 0.96 | 85.53 ± 0.93 |
| `--calendar --time-deltas` | 85.56 ± 1.09 | 86.59 ± 2.10 |

**The test gain is not in doubt.** +3.00 over the baseline, reproducing the gate's +2.96
almost exactly, with the spread five times tighter (0.21 against 1.08). The monotone trend
column adds nothing and triples the variance, so the safe option is also the better one.

**The validation comparison decides it, and it takes pairing to see that.** I first read
−1.92 against the summary sds (0.43 and 1.12) as ~1.6 SE and called it undecided. That is
the wrong denominator: seed *i* draws the same rows and the same model randomness in both
runs, so the per-seed differences are the statistic.

| seed | none val | cal val | Δval | none test | cal test | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 86.68 | 84.19 | −2.49 | 83.92 | 85.78 | +1.86 |
| 1 | 87.16 | 85.04 | −2.12 | 82.29 | 85.46 | +3.17 |
| 2 | 87.54 | 86.41 | −1.13 | 81.88 | 85.86 | +3.98 |

**Δval −1.91, sd 0.70, SE 0.41 → 4.7 SE. Δtest +3.00, sd 1.07, SE 0.62 → 4.9 SE.** Both
unanimous. Three replicates are ample once paired, and twelve would have been 1.5 hours
spent re-answering a settled question.

So **validation decisively rejects calendar and test decisively prefers it** — a sharp
anti-correlation, and the third independent demonstration of it on this task after recency
(+7.50, rejected 3/3) and the earlier categorical result (validation 81.63 vs 81.40 while
test said −3.52). **rel-event's validation split anti-predicts its test split**, and that is
now a property of the task rather than a run of bad luck.

Calendar is therefore the seventh effect here that is real on test and not selectable, and
the one with the most at stake: adoption would take rel-event from 80.98 to roughly 85.7,
past TabPFN-REL's 85.38 and within half a point of RelGNN's 86.18.

*Also worth noting: within the calendar family validation ranks correctly, putting
`--calendar --time-deltas` top, which is also the best on test. Only the calendar-versus-none
decision is inverted — and that is the one the replicate count cannot resolve.*

### 2026-08-05 — CORRECTION: the categorical blocks are not dead, they were measured on the wrong configuration

Earlier today I closed the categorical blocks on the basis of five gate measurements, the
largest being +0.27, and wrote that the question was "closed rather than open". Those were
all taken with **three child tables**. Re-run on rel-trial with **all ten**, 5 seeds:

| variant | gap | SE | positive | columns changed |
|---|---:|---:|:---:|---:|
| **`narrow`** (budget the nunique block) | **+0.99** | 0.10 | **5/5** | −38 |
| **`categories`** (top-K proportions) | **+0.90** | 0.12 | **5/5** | +144 |
| `booleans` | +0.38 | 0.05 | 5/5 | +26 |
| `timing` | +0.03 | 0.56 | 3/5 | +90 |

Both clear, both 5/5, both with standard errors under 0.13. The verdict was
configuration-dependent and I stated it as though it were not.

**`narrow` is the larger of the two and it *removes* columns.** `max_columns` has never
bounded the `nunique` block, so every categorical column emits a distinct-count however
narrow the budget is set; with three children that was 10 stray columns and worth nothing,
with ten children it is 38 and worth a point. The defect was correctly identified this
morning and its cost was measured on the configuration where it barely mattered.

It was also unreachable from the production runner until now — exposed as
`--budget-categoricals`.

**What this says about everything else closed today:** every gate in this log ran at
`--children 3`, which is both an arbitrary count *and*, as the entry below shows, an
environment-dependent set. Null results measured there bound nothing about the ten-child
configuration. The categorical entry below is left standing rather than deleted, because
what it says about three children remains true.

### 2026-08-05 — the prediction timestamp: +2.96 on rel-event, −0.68 on rel-trial

Both runners drop every datetime column when assembling the entity block, the cutoff
included, so nothing downstream could tell a Monday from a Saturday. Eight columns —
day-of-week, day, month, a weekend flag and sine/cosine pairs — plus a ninth for the
monotone trend, kept separate because every test row lies beyond the training range on it.

Gated paired by seed, 5 seeds, identical everything else:

| task | horizon | train span | `calendar` | `calendar-trend` |
|---|---:|---:|---:|---:|
| **rel-event** | 7 d | 147 d (≈21 weeks) | **+2.96** (SE 0.42, 5/5) | **+3.56** (SE 0.43, 5/5) |
| rel-avito | 4 d | **8 d** (≈1 week) | +0.09 (SE 0.13) | +0.17 (SE 0.25) |
| rel-trial | 365 d | 6,570 d | **−0.68** (SE 0.31, 1/5) | −0.54 (SE 0.22) |

**It is a short-horizon feature that also needs a long enough training span**, and rel-event
is the only task with both. rel-avito's horizon is short but its training window is barely
one week, so there is no weekly rhythm to learn; rel-trial's horizon is a year, over which
day-of-week is noise, and the eight columns dilute — it clears the floor *downwards*.

**Unlike recency, this has a real chance of being selected.** Recency's value grows with
distance from the training period, which is exactly what validation cannot see. A weekly
rhythm is stationary and validation has the same one test does. Promotion running.

**Also measured, and all null:**

| variant | rel-f1 | rel-event | rel-avito | rel-trial |
|---|---:|---:|---:|---:|
| `timing` (recency/age/span, 27 cols) | **+0.62** (SE 0.17, 5/5) | +0.91 (SE 0.73) | −0.57 (SE 0.28) | −0.60 (SE 0.42) |
| `narrow` (budget the nunique block) | n/a | +0.32 | n/a | +0.24 |
| `booleans` (rates instead of a nunique) | n/a | n/a | n/a | −0.24 |

**`timing` clears on rel-f1** — the tightest gate result of the day, 5/5 with SE 0.17 — and
is negative on rel-avito and rel-trial. The discriminator is what the child tables *are*:
race results and user activity are event streams, so "days since the last one" carries
signal, while rel-trial's `conditions`/`designs` are attributes recorded once at study
registration and rel-avito's 8-day span leaves nothing to be recent relative to. Promotion
running.

`booleans` fired for the first time here — 26 columns changed on rel-trial, so the
`'t'`/`'f'` detection works — and does nothing.

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

### 2026-08-06 — GATE: the task table's own label history, and it is the biggest signal found here

**We use the task table for three things — entity key, timestamp, label — and have never
used its fourth: the labels of that entity's EARLIER rows.** `eval_track_record` touches
`task.entity_col`, `task.target_col` and the time column and nothing else. Every feature in
this pipeline comes from the database's child tables. The task table is itself a
timestamped table keyed by entity, and its past rows are as legitimate a source of as-of
features as any child table — this is a gap in what we build, not a rule we were respecting.

Gated first, per the standing rule. For each training row, the feature is the mean of that
entity's strictly-earlier labels (`groupby(entity).shift(1).expanding().mean()` on rows
sorted by time), so no row can see its own label or any later one:

**The horizon correction, made before any of these numbers were acted on.** My first pass
used `shift(1)`, which requires only that the earlier row be earlier. That is wrong:
RelBench labels answer "does X happen within `task.timedelta` of this cutoff", so a label
recorded at *t* is not knowable until *t* + horizon. `key_target_history` already takes a
`label_horizon` for exactly this reason and I did not apply it. Redone as an as-of join on
the **resolved** time, `label_time + horizon`:

| task | coverage | AUC, no horizon (wrong) | **coverage** | **AUC, horizon applied** | ours |
|---|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 93.2% | 86.92 | 88.6% | **84.66** | 81.98 |
| rel-f1 / driver-dnf | 93.2% | 74.91 | 90.9% | **74.27** | 69.66 |
| rel-event / user-ignore | 55.7% | 82.94 | 46.4% | **81.72** | 80.98 |
| rel-event / user-repeat | 63.9% | 70.04 | 49.7% | 67.17 | 77.89 |
| rel-avito / user-visits | 48.1% | 57.32 | — | — | 65.54 |
| rel-avito / user-clicks | 41.8% | 59.06 | — | — | 65.89 |
| rel-trial / study-outcome | 0.0% | unavailable | 0.0% | unavailable | 72.26 |

**The error was small because RelBench spaces an entity's task rows exactly one horizon
apart** — median gap equals the horizon on all four, and 100% of gaps are at least the
horizon, so the previous row had almost always resolved. It cost 2.26 on driver-top3 and
0.64 on driver-dnf, mostly through coverage. The two rel-avito tasks were not redone: both
were already well below our pipeline, and the correction only moves numbers down.

**One claim did not survive it.** At 86.92 this feature looked to be above RelGNN's
field-best 85.69 on driver-top3; horizon-correct it is **84.66, below RelGNN**. The
surviving claim is narrower and still large: **on both rel-f1 tasks a single scalar beats
our entire pipeline** — by 2.68 on driver-top3 and 4.61 on driver-dnf, our worst placing in
the field at 9 of 10.

The structure is legible and predicts where it will work: **coverage tracks how often an
entity recurs.** Drivers race repeatedly (93%), users act repeatedly but sparsely (42–64%),
and a clinical trial has exactly one outcome (0.0%, `entities == rows`) — so rel-trial can
never benefit and needs no experiment.

**Why this is not yet a result.** These are AUCs over *training* rows, where the previous
label is typically one event old. At test time the label history freezes at the end of
train+val, so every test row is reading a staler history across a gap — the same temporal
mechanism that made `recency` real-on-test and unselectable. Persistence over that gap is
the whole question, and a train-row gate cannot answer it. Two further hazards to design
against before believing any number:

* **Coverage differs between train and test rows,** so a naive column is NaN-heavy exactly
  where it is least tested. Zero-filling is load-bearing here (see the keepnan entry), and
  coverage must be reported per split, not pooled.
* **The as-of construction must be by-construction, not by-assertion.** The prior finding
  is that leaks arrive as output shaped like a careful result. `assert_no_perfect_feature`
  will not catch a subtly-late label, since 86.92 is not 99.5.

Worth noting what it may also explain: RelGNN and RelGT — the two methods above us most
often — are graph models over a schema that includes the task table, so past labels can
reach a node through message passing without anyone designing a feature. Our two worst
placings are both rel-f1, and rel-f1 is where this signal is strongest.

### 2026-08-06 — DEFAULT CHANGED: `--children` 3 → 0, because 3 was not reproducible

`--children 3` was a hardcoded slice that "was never chosen". Counting what it actually does
across the benchmark settles it in one line:

| dataset | timestamped child tables | what `--children 3` did |
|---|---:|---|
| rel-avito | **3** | nothing — took all of them |
| rel-event | **3** | nothing — took all of them |
| rel-f1 | **3** | nothing — took all of them |
| rel-trial | **10** | discarded seven, and chose *which* seven non-deterministically |

**So the default only ever bit on one dataset, and there it was a reproducibility defect
rather than a tuning choice.** Two hosts running the identical command selected different
tables — `['designs','eligibilities','drop_withdrawals']` on one and
`['facilities_studies','sponsors_studies','designs']` on the other — because the slice
follows dictionary insertion order. The same command produced different features, which
makes every rel-trial comparison across machines untrustworthy unless both happened to agree.

**Changed to 0 (all tables).** On three of four datasets this is provably a no-op. On
rel-trial it removes the nondeterminism outright and is worth **+0.53** (72.79 against a
standing 72.26). The cost is compute on wide schemas, not accuracy, and `max_columns` already
bounds the per-child width. `STANDING_FLAGS` records `--children 3` for **rel-trial only** —
listing it for the other three would imply a difference that does not exist and send someone
looking for it.

**This is the second hardcoded default found to be arbitrary rather than chosen**, after
`max_columns=2`. Both were discovered by asking what the value actually does on each schema
rather than what it was supposed to do, and in both cases the answer was "nothing on most of
them, and something unintended on one".

### 2026-08-06 — ABSTENTION REFUTED, and it corrects my diagnosis of rel-avito

Tuning loses 1.29 on rel-avito/user-clicks (untuned `base` 67.32 against a calibrated 66.03),
which is four places in the published field. `--abstain` was built to detect that case
without consulting test: split validation by time, pick the winner on the early half, keep it
only if it still beats the untuned arm on the late half. The falsification condition was
recorded before the run — it had to fire on rel-avito and stay silent on rel-trial and
rel-event.

**It fired zero times in four seeds on user-clicks.** `+struct` won the early half *and* the
late half in every replicate:

```
ranking holds across the val gap (+struct: 64.12 early, 66.08 late) -- tuning kept
ranking holds across the val gap (+struct: 63.91 early, 66.80 late) -- tuning kept
ranking holds across the val gap (+struct: 63.94 early, 67.16 late) -- tuning kept
ranking holds across the val gap (+struct: 64.69 early, 67.30 late) -- tuning kept
```

Result 66.18 ± 0.61 against the ordinary 66.03 ± 0.54 — a no-op, and the 0.15 is protocol
noise, not an effect.

**THIS CORRECTS THE DIAGNOSIS I GAVE TWO ENTRIES ABOVE.** I wrote that rel-avito loses
because "the arms are near-identical and selection is fitting noise on a small validation
split". That is wrong on both counts. Validation is **21,183 rows** — not small — and its
preference is **not noisy**: it picks `+struct` over `base` consistently, in every seed, on
both halves of its own time range. Test then prefers `base` by 1.15. So validation is stable
and systematically pointing the wrong way. **Bias, not noise** — the same finding that closed
the selection thread, showing up on a task where I had misattributed it.

**And that is why abstention cannot work.** The instrument checks validation against
*itself*. Both halves of validation share whatever makes validation disagree with test, so a
consistent bias is exactly the thing this design is blind to — it can only catch an
*unstable* ranking, and the ranking here is perfectly stable. **Any instrument that validates
validation internally is refuted by this result, not just this one.** That is a larger class
than `--abstain`: it covers held-out-validation splits, val-internal cross-validation, and
ranking-stability checks generally.

**What is not explained.** The late half of validation scores 66–67, the same range as test,
so this is not simple temporal distance — a *later* validation slice at test-like difficulty
still prefers the arm test rejects.

**I proposed entity novelty as the mechanism, measured it on all seven tasks, and it is
refuted.** No model needed: how many query entities were seen in train, on val versus test.

| task | tuning | val seen | test seen | **drop** | base-rate shift |
|---|---:|---:|---:|---:|---:|
| rel-trial / study-outcome | +2.74 | 0.0% | 0.0% | 0.0 | 0% |
| rel-event / user-ignore | +1.76 | 69.8% | 68.1% | −1.8 | 18% |
| **rel-f1 / driver-dnf** | **+1.48** | 69.6% | 35.2% | **−34.4** | 10% |
| rel-event / user-repeat | +0.76 | 73.5% | 68.7% | −4.8 | 8% |
| rel-f1 / driver-top3 | +0.63 | 67.2% | 35.3% | −31.9 | 13% |
| rel-avito / user-visits | −0.11 | 81.2% | 58.6% | −22.6 | 6% |
| **rel-avito / user-clicks** | **−1.29** | 73.5% | 39.4% | **−34.0** | **56%** |

**`driver-dnf` loses 34.4 points of entity overlap between val and test — the same as
user-clicks — and tuning gains +1.48 there.** One counterexample of that size on seven tasks
kills it. The r = +0.63 is carried by rel-trial sitting at 0/0, not by the hypothesis.

**The base-rate shift correlates better (r = −0.73) and I am not going to claim it.**
user-clicks' positive rate more than halves from validation to test — **0.035 → 0.015, a 56%
shift against a next-highest of 18%** — and it is the one task where tuning clearly loses.
With n = 7 and a single point that far from the rest, removing it collapses the correlation.
That is a description of one task, not a law.

**And there is a reason not to chase it further: the predictive signal is unusable and the
usable signal is unpredictive.** Computing a test base rate requires test labels, which is
exactly what the protocol withholds — so even a perfect relationship there could explain the
failure and never decide anything. Entity overlap *is* label-free and legitimately
computable at inference time, and it is the one that does not predict. Any rule for when not
to tune has to be built from quantities in the second column, and this says the obvious
candidate there is not it.

**The full sweep, all seven tasks, both arms paired:**

| task | ordinary | `--abstain` | Δ | vetoes fired |
|---|---:|---:|---:|---:|
| rel-avito / user-clicks | 66.03 | 66.18 | +0.15 | **0** |
| rel-avito / user-visits | 65.50 | 65.50 | +0.00 | **0** |
| rel-trial / study-outcome | 72.30 | 72.19 | −0.11 | **0** |
| rel-event / user-ignore | 81.98 | 81.28 | −0.70 | **0** |
| rel-event / user-repeat | 77.89 | 77.83 | −0.06 | **2** |
| rel-f1 / driver-top3 | 82.13 | 82.02 | −0.11 | **0** |
| rel-f1 / driver-dnf | 68.90 | 69.22 | +0.32 | **0** |

**Two vetoes across roughly forty selection decisions, and both landed on the wrong task.**
They fired on rel-event/user-repeat — where tuning is worth **+0.76** and there is nothing to
abstain from — and never once on either rel-avito task, where tuning loses 1.29 and 0.11.
That is the falsification condition failing in both directions at once: silent where it was
needed, active where it was not.

**The deltas are not the veto's doing.** With the veto firing zero times on five of seven
tasks, those arms still differ, because the implementation was also replacing the pick with
the early-half winner instead of the full-validation argmax — two variables, not one. That is
fixed (the flag is now a pure veto), and it incidentally priced the confound: spending 40% of
validation on a meta-decision costs **mean −0.17, worst −0.70** on rel-event, which is above
the ±0.6 floor. **Any future instrument that carves up validation to decide something pays
that toll before it delivers anything**, which raises the bar for the whole category.

**Kept, off by default, as a recorded negative.** Correct code, five tests, free to run, and
blind to the failure it was aimed at.

### 2026-08-06 — categories across all seven tasks: positive wherever it exists, absent on four

Categories was written off here once on a `--children 3` measurement, then cleared at ten
children on rel-trial (+0.90), then reached +0.96 paired on rel-event/user-repeat with
validation agreeing in sign. Run across every task, test-side, uncalibrated arms:

| task | without | with `--categories 8` | Δ on `base` | other arms |
|---|---:|---:|---:|---|
| rel-event / user-repeat | 77.13 | 78.49 | **+1.36** | +struct +1.29, +counts +1.45 |
| rel-event / user-ignore | 80.22 | 81.24 | **+1.02** | +struct +0.75, +counts +0.40 |
| rel-trial / study-outcome | 69.95 | 70.37 | +0.42 | +counts +0.21, +rate +0.50 |
| rel-avito / user-visits | 65.61 | — | — | **no categorical columns** |
| rel-avito / user-clicks | 67.32 | — | — | **no categorical columns** |
| rel-f1 / driver-top3 | 82.19 | — | — | **no categorical columns** |
| rel-f1 / driver-dnf | 69.19 | — | — | **no categorical columns** |

**Positive on every arm of every task where it applies — nine of nine — and structurally
absent on four of seven.** On those four the `--category-share 0.5` gate rejects every
categorical column as free text and the empty-block guard refuses the run rather than
reporting the +0.00 that an empty block would otherwise produce. That refusal is a result
about the schema, not a failure.

**Settled on validation, as a default must be — and the answer is less clean than the
test-side table.** Paired per seed on the `base` arm at matched context, under the calibrated
protocol:

| task | **validation** Δ | SE | positive | calibrated **test** Δ |
|---|---:|---:|---:|---:|
| rel-trial / study-outcome | **+1.12** | 0.22 | 21/24 | +0.60 |
| rel-event / user-repeat | +0.25 | 0.07 | 24/36 | **+0.93** |
| rel-event / user-ignore | +0.26 | 0.18 | 13/18 | **−0.44** |

**Validation is positive on all three and negative on none, so the worst-case regret of
switching categories on — measured the way `max_columns=4` was — is zero.** That is a
legitimate basis for a default, and it is the only evidence allowed to choose one.

**But test disagrees on rel-event/user-ignore, and I am not going to hide behind the
procedure.** Validation says +0.26 there; the calibrated test result is **−0.44**. Note also
that categories *helped* that task's plain arms (+1.02 on `base`, +0.75 on `+struct`) and
still lost once selection was in the loop — the wider grid changed what validation picked,
for the worse. Across the three applicable tasks the test mean is **+0.36**; across all seven
it is **+0.16**, since four cannot use it at all.

**So the honest position is that the evidence is mixed, and the procedure that says
"switch it on" is the same procedure this file has spent the session showing is
systematically wrong on this benchmark.** Choosing a user-facing default on the weaker signal
purely because it is the admissible one would be following the letter of the discipline
against its point. **Recommended but not flipped; this is the maintainer's call**, and it
should be made knowing that one of three measurable tasks gets worse.

**user-repeat is confirmed at 12 replicates**: +0.93 test (against +0.96 at 8), so that one
is solid rather than a lucky draw. **rel-trial's 73.39** is its best number recorded here,
but it is `--children 0 --categories 8` — the child-table change is worth +0.53 of it and
categories +0.60, so it is not a categories result alone and does not belong in the headline
table as one.

**A design consequence that holds regardless of how that lands.** The empty-block guard
currently **aborts** the run, which is right for an explicitly requested block — an empty one
silently scoring +0.00 is how depth-2 was "measured at no effect" for a week. It is wrong for
a *default*: a user on a rel-avito-shaped schema would get a crash instead of a model. If
categories becomes a default, the guard has to distinguish "you asked for this and it is
empty" (abort) from "this is on by default and does not apply here" (skip, and say so).

### 2026-08-06 — a leakage control that fired on nothing, and cost rel-avito its only user-similarity feature

**rel-avito does build the User → Ad → User similarity feature, and then throws it away.**
The candidate keys there are `PhoneRequestsStream_AdID`, `VisitStream_AdID`,
`SearchInfo_LocationID`, `SearchInfo_CategoryID`, and `key_target_history` turns each into
`n_prior`, `n_positive`, `positive_rate`, `n_linked` as of the row's cutoff. So
`VisitStream_AdID__positive_rate` is exactly "the outcome rate among other users who visited
the same ad" — a target-encoded historical statistic over similar users, membership-weighted
by how many keys two users share. It is computed on every run. And on rel-avito:

```
arm eligibility: {'base': True, '+struct': True, '+counts': True,
                  '+rate': False, '+history': False, '+text+rate': False}
  +rate is EXCLUDED from selection: its own controls failed
```

**The control that excluded it was testing nothing.** The temporal control rewinds every
cutoff and checks the score does not improve, with shifts scaled to the task's span:
`round(0.05 * span)` and `round(0.15 * span)`. rel-avito's train span is **8 days**, so
those are **0 and 1**.

* The first "control" was **shift 0 — the unshifted setting itself.** That is why the report
  printed `control=0.5645`: it is the mean of the observed 0.5618 and the single real
  measurement 0.5671, not a second control.
* The only real shift was **1 day against a 4-day label horizon.** An outcome only becomes
  readable one horizon after it is recorded, so a 1-day rewind cannot withhold one. The
  printed coverage says so plainly: **0.572 → 0.571.**
* On that basis a **0.0053** difference was reported as *"the features are reaching past the
  cutoff"* and the whole `+rate` family was barred from selection on this dataset.

**rel-trial had the same defect more quietly**: a 150-day first shift under a 365-day
horizon. rel-event (7, 22 against a 7-day horizon) and rel-f1 were always fine, which is why
this survived — the two tasks with long spans looked healthy.

**Fixed by flooring every shift at one horizon, dropping zeros, and deduplicating** — two
fractions that round to the same number are one control, not two. Extracted as
`temporal_shift_grid` with five tests, two of which pin rel-event at `(7, 22)` and rel-f1 at
`(1000, 3000)` **unchanged**, so a fix to the broken tasks cannot silently re-open verdicts
reached on the healthy ones. Added the mirror of the existing vacuity guard: a shift moving
coverage by under 1% now says outright that a LEAK verdict from it is an artefact of the
fill value rather than evidence.

**Thirteenth defect, and the worst-shaped one yet.** The earlier twelve produced numbers
that looked like results. This one produced a *refusal* that looked like rigour — a control
failing is exactly what a careful pipeline is supposed to do, so it reads as the system
working. A false negative from a safety check is harder to see than a false positive from a
measurement, because nobody audits the thing that said no.

**RESOLVED, and not in the direction I expected: `+rate` still fails, and now legitimately.**
Re-run with the repaired grid — `shifts (4,)d` against the 4-day horizon —
rel-avito/user-clicks reports the same verdict, and the new indeterminacy note **did not
fire**, meaning the shift moved at least 5% of feature values. So the control was
informative and it failed. The exclusion of the User → Ad → User target encoding on that task
stands on evidence.

**Both things are true and the distinction matters.** The old grid *was* broken — it rewound
1 day against a 4-day horizon and its verdict meant nothing — so fixing it was necessary.
But I wrote that the verdict was "an artefact", and a properly-powered control reached the
same conclusion. The right reading of the original entry is *"this verdict was unfounded"*,
not *"this feature is fine"*. A broken instrument that happens to agree with a working one
was still broken, and a working one that agrees with it was still worth building.

**And on rel-avito/user-visits every arm now passes**, `+rate` and `+history` included — so
the repaired control is not simply stricter, it discriminates between the two tasks. On that
task `+rate` at **65.85** is the best block available, above base 65.61 and above the
calibrated 65.50, which the selection rule declined to take.

**What is still not known:** whether `+rate` is worth anything on user-clicks, since it
remains excluded there and cannot be scored inside the protocol. Eligibility is permission to be
selected, not evidence of value — `+rate` scored **0.5618** standalone against a 65.54
pipeline, so promoting it into the selection pool could make the calibrated number *worse*.
Both are measured in the next round. Meanwhile the two arms that *are* eligible on rel-avito
carry the label-free half of the same idea: `n_linked`, the co-visitation degree, at 61.43
standalone, and the resolved-label counts at 61.83.

**Still absent from this pipeline on rel-avito**, for the record: no K-NN or top-K cut over
neighbours, no clustering or distance to cluster centres, no embedding — similarity is exact
shared-key co-occurrence. And no message passing: the 2-hop path exists as a join, but the
aggregation is an unweighted mean, with no attention weighting neighbours dynamically.
`neighbour_label_features` and `select_graph_context` exist but were built against
rel-event's explicit friendship edges; rel-avito has no user–user table.

### 2026-08-06 — categories on user-repeat: the first effect validation agrees with

Seven effects here are real-on-test and unselectable, because validation sits near train and
test sits far. `--categories 8` on rel-event/user-repeat is the first candidate in a long
while where **validation moves in the same direction as test**. Both arms calibrated,
8 replicates, paired by seed:

| | defaults | `--categories 8` | paired Δ | SE | t | positive |
|---|---:|---:|---:|---:|---:|---:|
| **validation** | 72.45 | 72.65 | **+0.20** | 0.15 | +1.34 | 5/8 |
| **test** | 77.89 | 78.85 | **+0.96** | 0.24 | **+4.05** | 7/8 |

Compare calendar, the archetype of the unselectable class: Δtest **+3.00** against Δval
**−1.91**. Here the signs agree. Test is decisive at t = 4.05 and well outside the ±0.6
floor; validation is positive but weak.

**What the selection rule actually delivers, simulated per seed rather than assumed.**
Choosing per replicate by validation score:

* always defaults → 77.89
* **selected on validation → 78.37** (validation picked categories 5/8)
* always categories → 78.85

**+0.48 over standing, at paired SE 0.24 — and selection leaves exactly half the available
+0.96 on the table.** That is the honest claimable number under this project's rule, and it
is a *weaker* claim than the test-side +0.96 that would be tempting to quote.

**Which points at the better move: make it a default, not a selection.** A validation signal
this weak (t = 1.34) will keep discarding half the effect. `max_columns` was settled the same
way — chosen across tasks by worst-case regret rather than per task — and that is the
procedure `eval_defaults.py` already implements. Categories has now cleared on **two**
schemas (rel-trial +0.90 at ten children, user-repeat +0.96 here) after being written off
once on a `--children 3` measurement. **Next: run categories through the worst-case-regret
default selection across all seven tasks.** If it is never much worse and sometimes +1, it
belongs on by default and no selection rule is needed.

**Not yet in the headline table.** 78.37 would still be 3rd of 10 on this task (RelGNN 79.61,
KumoRFMv2 79.34), so no rank changes either way; the number goes in when the default question
is settled, not before. One caveat travels with it: `--categories 8` produced **no columns at
all** on rel-f1/driver-dnf and the empty-block guard refused the run rather than reporting a
null — so "on by default" cannot mean "assume it emits something".

### 2026-08-06 — IDEA 1 REFUTED: a feature that beats the pipeline standalone adds nothing

Label history was the largest gate result in this project: standalone **test** AUC 85.35 on
rel-event/user-ignore against a full pipeline at 80.98, and it survived the horizon
correction and the train-versus-test coverage check that killed its rel-f1 numbers. Paired
A/B, same seeds, one flag apart:

| task | arm | without | with | **paired Δ** | paired SE | t | positive |
|---|---|---:|---:|---:|---:|---:|---:|
| user-ignore | `base` | 80.22 | 81.12 | **+0.90** | 0.44 | +2.03 | 5/6 |
| user-ignore | `+struct` | 82.09 | 82.17 | +0.08 | 0.35 | +0.24 | 3/6 |
| user-ignore | `+counts` | 81.15 | 81.36 | +0.21 | 0.30 | +0.70 | 4/6 |
| user-repeat | `base` | 77.13 | 76.51 | **−0.62** | 0.14 | **−4.40** | 1/8 |
| user-repeat | `+struct` | 77.33 | 76.61 | **−0.71** | 0.18 | **−4.03** | 1/8 |
| user-repeat | `+counts` | 77.21 | 76.57 | **−0.64** | 0.18 | **−3.56** | 0/8 |

Calibrated: user-ignore 81.98 → 82.28 (**+0.30**), user-repeat 77.89 → 77.07 (**−0.82**).

**On user-repeat it hurts, and that verdict is not close** — three arms agree, all at t ≈ −4,
positive on 1 of 8, 1 of 8 and 0 of 8. On user-ignore it helps only on `base` and does
nothing on `+struct`, which is the arm the calibrated protocol actually selects.

**The control confirms it from the opposite direction.** rel-f1/driver-dnf was included as a
control because its test-side gate was the *weakest* of the three — 65.20 standalone against
a 69.66 pipeline, 35% coverage, and a median staleness of 2,486 days. It is the only one of
the three where label history **helps**: `base` 69.19 → 69.98 (**+0.79**), calibrated 68.90
→ 69.24 (+0.34, 5 replicates, inside the floor).

| task | standalone test AUC | Δ on `base` | Δ on the selected arm |
|---|---:|---:|---:|
| rel-event / user-ignore | **85.35** (best) | +0.90 | +0.08 |
| rel-event / user-repeat | 78.61 | −0.62 | **−0.71** |
| rel-f1 / driver-dnf | **65.20** (worst) | **+0.79** | +0.34 |

**The ordering by standalone AUC is the reverse of the ordering by contribution.** Best
standalone contributes nothing; worst standalone contributes most.

**The mechanism is redundancy, and it is legible.** `+struct` is the `key_target_history`
block — other entities' outcomes reached through a shared key, plus structural counts. Where
that block is present, label history adds +0.08; where it is absent (`base`), it adds +0.90.
And **rel-f1 has no candidate keys at all** — its `means:` line carries only a `base` arm,
because there is no shared-key track record to build. That is exactly the schema where this
feature is not redundant, and it is exactly the schema where it helps. Three tasks, one
mechanism, no exceptions.

**THE LESSON, and it qualifies the rule this whole project runs on.** "Gate before you
build" has been the standing discipline here and it is still right, but a gate measures
**whether a feature carries signal**, not **whether that signal is new**. Those are
different quantities and this is the cleanest possible demonstration: 85.35 standalone —
better than our entire pipeline, better than eight of the ten published methods on that task
— converting to **+0.08** on the arm that gets selected. Marginal contribution is the
quantity that matters and only a paired A/B against the real feature set measures it.
Nothing about the earlier gates was wrong; they simply cannot answer this.

**Kept, not reverted, and now with a rule for when to reach for it.** `entity_label_history`
and `--label-history` stay in the tree, off by default. The construction is correct, the
tests pin the self-exclusion guarantee, and the three tasks agree on when it earns its
place: **use it when the schema has no shared foreign key to build a track record from, and
skip it when `key_target_history` already applies.** rel-f1 is the first case, rel-event the
second, and rel-trial cannot use it at all. Nothing here is table-eligible: +0.34 on
driver-dnf at 5 replicates is inside the ±0.6 floor and the calibrated protocol did not
select it.

**One number in the earlier entries is now known to be a poor predictor and stays on the
page anyway.** The 85.35 that motivated all of this is a correct measurement of the wrong
quantity, and leaving it visible with the outcome attached is more useful than editing it
down to look prescient.

### 2026-08-06 — best-cfg for rel-avito/user-clicks, and two tasks I threw away

A 21-configuration sweep per task — defaults, context 1000, all children, categories,
top-keys, calendar, resample, each also covering the three frame variants — run to fill the
three empty `best-cfg` cells. One task's worth survived:

**rel-avito/user-clicks: best-cfg 68.19** (`--resample 4`, `base` arm), from 20 of 21
configurations. Standing is 65.89, so the labelled upper bound is **+2.30** above what the
calibrated protocol actually selects, and 68.19 would still sit below RDBLearn+v3's 69.06.
Top five: 68.19, 67.49, 67.15, 66.96, 66.87 — a 1.32 spread across the top five and 3.93
across all twenty.

**rel-f1/driver-dnf: at least 69.95**, from only 2 of 21 configurations, so it is a lower
bound on an upper bound and goes in the table with that count attached, or not at all.

**rel-event/user-repeat: nothing. I destroyed it.** The launcher piped a multi-hour run
through `tail -300`, which kept the last 300 lines and discarded the first two tasks before
anyone read them. The sweep itself ran correctly and the pod had already been terminated by
the time the loss was visible, so the data is gone rather than recoverable.

**This is the fifth time an output filter has eaten results in this project, and the first
time I did it after writing the other four up.** The earlier four were `tail -60` and a
`keep` grep discarding per-seed data. The fix is not another reminder: `remote.sh` now
`tee`s the run to `/workspace/work.log` on the pod, so a truncated local log is survivable
as long as the pod is still up, and the launcher no longer pipes through `tail` at all.

**A NEW LEAD, and the strongest thing in this round: `--categories 8` on
rel-event/user-repeat.** Its three arms score 78.49 / 78.58 / 78.56 against 77.07 / 77.26 /
77.11 at defaults — **about +1.4, consistent across all three arms**, comfortably outside the
±0.6 floor, and it supplies the whole of that task's best-cfg cell. Categorical blocks were
written off here once already on a `--children 3` measurement and then cleared at ten
children on rel-trial (+0.90), so this is the second schema where they matter and the first
where the margin is this size. It is a **test-side** number, which on this benchmark is the
beginning of the question rather than the end — seven effects are already real-on-test and
unselectable. The next run should be a calibrated one on user-repeat with categories in the
grid, to find out which of those two things this is.

**The parser I wrote to avoid transcription errors made a worse one.** It read the reference
suffix on rel-event's output lines — `(ours 77.89, TabPFN-REL 77.11, RelGNN 79.61, ...)` —
as though those were our own configurations, and reported **BEST-CFG 79.61**, which is
RelGNN's published score, as ours. It was caught only because the winning "configuration"
was named `topkeys/RelGNN`. A competitor's number that did not carry a method name in the
label would have passed straight into the table. Fixed by stripping everything from the
first `(`. Twelfth defect in this project, and the second where a tool built to enforce
care was itself the thing that was careless.

**A note on what `best-cfg` is worth as a number.** It is a max over 21 noisy scores, and a
max over K noisy draws is biased upward with the bias growing in K. The original four tasks'
cells came from every variant ever run on them — far more than 21 — so those cells are
*more* inflated than these, not less. The column is labelled an upper bound selected with
knowledge of test for exactly this reason, and the configuration count now travels with each
cell so the comparison between them is at least visible.

### 2026-08-06 — IDEA 2, from data already collected: per-task calibration buys nothing

Every instrument built from this validation split has failed, and the diagnosis was **bias,
not noise**. The move that remains is not a better instrument but **not selecting at all**:
fix global defaults by worst-case regret and run them everywhere. Four tasks already have
both numbers — the untuned `base` arm and the validation-selected one — so this much needs
no GPU:

| task | untuned `base` | calibrated | calibration is worth |
|---|---:|---:|---:|
| rel-event / user-ignore | 80.80 | 80.98 | +0.18 |
| rel-avito / user-visits | 65.81 | 65.54 | −0.27 |
| rel-avito / user-clicks | 67.18 | 65.89 | **−1.29** |
| rel-f1 / driver-dnf | 68.99 | 69.66 | +0.67 |

**Mean −0.18, negative on 2 of 4, and 2 of 4 inside the ±0.6 floor.** Selecting a
configuration per task on validation is worth nothing on average and costs 1.29 on
user-clicks, which is four places in the field. That is the central finding showing up as a
protocol recommendation rather than a diagnosis.

**Two tasks are missing and both are the interesting kind.** rel-f1/driver-top3 (+8.48) and
rel-trial (+2.64) are the cases where calibration looks decisive — and on rel-f1 almost all
of it was the **column budget**, which the shipped `max_columns=2` got wrong for that schema
by 12.58 on validation. That is not per-task selection earning its keep; it is a bad global
default being repaired per task.

**Which is why this measurement is not finished, and the reason is a change we made.** All
six figures were taken at `max_columns=2`. The default is now **4**, chosen on validation
across four tasks by worst-case regret: never more than 2.13 off the best value for a task,
against 19.13 for the old 2. If the case for calibration was mostly budget repair, then
fixing the default should collapse it — the honest test is untuned-at-4 against calibrated,
paired, on all seven tasks. `work18.sh` collects the `base` arm from the same invocations
it uses for label history, so three of the seven come free.

**Predicted, before the run:** calibration's advantage on rel-f1/driver-top3 largely
disappears, and the seven-task mean lands at or below zero. If instead it survives at 4,
then per-task selection is doing something the regret analysis missed, and the central
finding needs revisiting rather than restating.

**RESOLVED — the prediction was right on the task it was made for and wrong elsewhere.** On
rel-f1/driver-top3, the task whose +8.48 motivated the whole idea, calibration with the
budget already correct is worth **−0.06**: the effect collapses entirely and it *was* budget
repair. On the rel-event tasks, where the budget was never the problem, calibration is worth
**more** at the better default, not less. See the rebuilt expectations table above. The
paragraphs below are what was written before driver-top3 landed and are kept as the record.

**THE PREDICTION IS WRONG so far, on both tasks measured at the new default.** At
`max_columns=4`, calibration is worth **+1.76** on rel-event/user-ignore (untuned `base`
80.22 against calibrated 81.98) and **+0.76** on user-repeat (77.13 against 77.89). At
`max_columns=2` the same comparison on user-ignore was +0.18. Selection got *more* valuable
when the default improved, not less.

The reason is visible in the arms: the untuned `base` number barely moved (80.80 → 80.22)
while the calibrated number rose a full point (80.98 → 81.98). A wider column budget did not
make the default configuration better — it enlarged the gap between the default arm and the
best arm, and selection is what captures that.

**rel-f1/driver-dnf, measured in the same round, goes the other way: −0.29** (untuned `base`
69.19 against calibrated 68.90). Three tasks at current defaults now read **+1.76, +0.76,
−0.29 — mean +0.74**, against −0.18 for the four measured at `max_columns=2`.

So "fix the defaults and stop selecting" is **not** supported, and the recorded prediction
should be read as refuted rather than quietly dropped. The honest current statement is
narrower than either the prediction or its opposite: **per-task calibration is worth roughly
+0.7 on average with the sign flipping by task, and it is worth more at a better default,
not less.** rel-avito and rel-trial at `max_columns=4` are outstanding and both were
*negative* at the old default, so the mean can still move. What is already clear is that the
mechanism I proposed — calibration as budget repair — is not what is happening.

### 2026-08-06 — label history on TEST: the gate ranked the tasks backwards

The gate above measured this feature over **training** rows, where an entity's previous
outcome is one event old. At test the pool freezes at the end of train and every query
reads across a widening gap. I recorded that as the hazard that decides the feature. It
did, and it inverted the ranking — the two tasks the gate liked most are the two it was
most wrong about. Pool is the training split, horizon applied, scored on test:

| task | train-row gate | test coverage | **test AUC** | pool=train+val | our pipeline |
|---|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 84.66 | 35.3% | **56.89** | 59.42 | 81.98 |
| rel-f1 / driver-dnf | 74.27 | 35.2% | **65.20** | 67.55 | 69.66 |
| rel-event / user-ignore | 81.72 | 68.1% | **85.35** | 83.24 | 80.98 |
| rel-event / user-repeat | 67.17 | 68.7% | **78.61** | 79.28 | 77.89 |

**Mechanism, measured rather than guessed — and it is worse on rel-f1 than "a later
season".** Taking `days_since` (cutoff minus the time the most recent prior outcome became
readable) over the covered test rows:

| task | covered test rows | median `days_since` | range |
|---|---:|---:|---|
| rel-event / user-ignore | 1,333 | **8 days** | 8 – 113 |
| rel-f1 / driver-dnf | 247 | **2,486 days** | 1,946 – 3,686 |

**rel-f1's covered test rows are reading labels five to ten YEARS old.** The pool freezes at
the end of train and rel-f1's test period is long, so even a driver who does appear in train
is being scored against a decade-stale record — and two thirds of test drivers do not appear
in train at all. rel-event's median staleness is 8 days, one horizon, which is as fresh as
this construction permits. That is the whole difference, and it is a property of how far the
test window runs from the frozen pool, not of the schemas.

**The decay runs the other way on rel-event, and the reason is not leakage.** AUC over the
covered rows is 83.63 in the fresher quartile and **99.47** in the stalest (22–113 days,
n=192). A near-perfect subgroup is normally a red flag, so: a user whose last recorded
outcome is months old is a user who went inactive, an inactive user ignores invitations, and
they were ignoring them before too. `days_since` alone scores **35.57** — informative
inverted, i.e. 64.43 the right way round — which is why it is emitted as its own column
rather than folded into the rate.

**This is why a train-row gate cannot promote a temporal feature.** Nothing about the gate
was wrong as a measurement; it answered a question about training rows, and the question
that matters is about a gap that training rows never see. Two tasks' worth of pod time was
about to be spent on the rel-f1 pair on the strength of it.

**Do not read 85.35 against 80.98 as a 4.37 gain.** The standalone AUC is over the **68.1%
of rows that have any history**, and the pipeline's 80.98 is over all of them. Comparing
scores computed on different row sets is the error this file has caught before. What the
pair licenses is that the feature is strong *where it applies* on this task, and nothing
about the combination — which is what the paired A/B measures.

**Consequences for what to run.** rel-event is where this belongs; rel-f1 is demoted to a
control, kept because a feature that is weak-but-not-absent may still cost something when
added. rel-trial remains excluded by construction (coverage 0.0%). Pool stays `train` —
the fitting split — even though `train+val` is better on user-repeat, because the choice
that varies by task is a selection decision and this file's central finding is that
selection on this benchmark is biased.

### 2026-08-06 — the backbone hypothesis, weakened by a ladder already in the table

The open question below asks whether our gap to the flatten-then-TFM peers is featurization
or backbone: they all run TabPFN-3, we run TabICL, and nothing here separates the two. I
built `eval_backbone.py` to settle it by swapping backbones on identical features — and it
is **blocked**: `tabpfn` 8.2.0 will not download weights without a `TABPFN_TOKEN` from a
registered priorlabs.ai account with the license accepted. The 2.x line (through 2.2.1,
Sept 2025) is ungated, but that is TabPFN v2 — a generation *below* what every peer uses,
and its 100-feature limit sits under our frame width, so it would measure extrapolation.
The runner is committed and needs only a token.

**But Table 14 already contains a backbone ladder on fixed features, and I had not read it
as one.** RDBLearn appears three times — plain, `+v2.5`, `+v3` — same featurization, three
backbone generations. That isolates the backbone axis directly:

| task | RDBLearn | +v2.5 | +v3 | v2.5→v3 | ours |
|---|---:|---:|---:|---:|---:|
| rel-event / user-repeat | 75.04 | 75.55 | 76.81 | **+1.26** | 77.89 |
| rel-trial / study-outcome | 71.58 | 72.90 | 72.89 | −0.01 | 72.26 |
| rel-f1 / driver-top3 | 79.69 | 77.60 | 82.72 | **+5.12** | 81.98 |
| rel-event / user-ignore | 82.52 | 78.65 | 73.70 | **−4.95** | 80.98 |
| rel-avito / user-visits | 65.49 | 66.47 | 66.76 | +0.29 | 65.54 |
| rel-avito / user-clicks | 69.04 | 65.72 | 69.06 | **+3.34** | 65.89 |
| rel-f1 / driver-dnf | 70.87 | 71.72 | 71.72 | 0.00 | 69.66 |
| **average** | **73.46** | **72.66** | **73.38** | **+0.72** | **73.46** |

**A full backbone generation, on fixed features, is worth +0.72 on average — sd 3.16,
positive on 4 of 7, and it ranges from −4.95 to +5.12.** Our gap to the best average is
2.60. So the backbone axis, as far as published evidence reaches, is *smaller than our gap
and mostly noise around zero*.

Two details sharpen it:

* **On rel-event/user-ignore the newer backbone made RDBLearn 4.95 WORSE.** That is the
  task where the backbone story was most tempting — we sit 4.40 below TabPFN-REL there. The
  one published measurement of a backbone upgrade on that exact task points the wrong way.
* **Plain RDBLearn averages 73.46, above its own `+v3` at 73.38 and its `+v2.5` at 72.66.**
  The base configuration beats both upgrades on average. "Newer backbone is better" is not
  supported even within one authors' own three rows.

**What this does and does not license.** It bounds *a generation step inside the TabPFN
family*, not *TabICL versus TabPFN*, which is a different-family comparison and could be
larger. It is also seven tasks of published single numbers with no replicate counts, so
none of these deltas is paired and the ±0.6 floor here is unknown and probably worse. It
does not close the question — `eval_backbone.py` still should run. What it does is move the
prior: **the expected value of backbone work is now around +0.7 with a large variance, and
that is not where a 2.60 average deficit is hiding.** Featurization remains the better bet,
and this is the second time the peers' published spread has turned out to be dominated by
something other than model quality — see the 11.68-point user-ignore entry above.

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
