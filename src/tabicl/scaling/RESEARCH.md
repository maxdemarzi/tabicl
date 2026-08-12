# Research directions

Candidate directions, ordered by expected value rather than by expected benefit.
`TODO.md` is the open engineering work; this is what to build after it.

Directions 1-6 came from a proposed research table; 7-12 were generated afterwards, and
**8 and 7 outrank most of the original list** -- 8 because it attacks the i.i.d.
limitation with no weight changes, 7 because it gates every label-derived feature here.

## The criterion used to order them

Across this branch, feature-level levers on RelBench landed almost entirely inside a
**±0.6 ROC-AUC noise floor** (measured: paired-gap sd 0.29 on rel-event), and the ones
that looked larger repeatedly reversed under a second measurement — look-back windows
(+0.089 on one task, median +0.001 elsewhere), a "2.5-point categorical gap" that turned
out to be a three-variable comparison, relation breadth (+0.35, inside the floor). What
held up under direct attack was infrastructure: row chunking is exact to 1.1e-05, the
memory guard, the correctness fixes.

So the ordering criterion is **not** expected accuracy gain. It is:

> Does this direction compete on a metric we have shown is hard to move, or does it
> deliver a capability that is not measured in AUC at all?

Directions of the second kind cannot be washed out by the noise floor. Directions of the
first kind need a plan for beating it before they are worth starting.

---

## 1. Sample attribution — exact row influence and calibrated uncertainty

**Bottleneck:** black-box predictions.
**Why first:** it is the one direction where in-context learning has a *structural*
advantage rather than a marginal one, and it cannot be defeated by the noise floor
because it is not an accuracy claim.

A trained model needs influence functions or leave-one-out retraining to answer "which
training rows caused this prediction". TabICL attends from each test row directly to
labelled context rows, so that information is present in the forward pass. The KV-cache
machinery already in this package (`kv_cache.py`, `_mqa.py`) is the right substrate:
context K/V are already materialised per block.

Scope:

* attention-based per-row influence, validated against leave-one-out on a small task —
  the validation is the work, the extraction is nearly free
* conformal prediction intervals for the uncertainty half; standard, cheap, and gives
  coverage guarantees rather than a softmax
* both compose with the context-selection work: if influence is concentrated in few
  rows, that is direct evidence about how small a context can be

**Risk:** attention weights are a proxy for influence, not influence. If the
leave-one-out validation fails the honest outcome is a negative result about attention
attribution, which is still worth publishing internally.

## 2. RAG-TabICL — retrieval-conditioned context

> **2026-08-05: the crudest possible version of this is the largest unclaimed effect in the
> project.** Selecting context by *recency* rather than at random is +7.50 on rel-event at
> a 1,000-row context (SE 0.96, 5/5 seeds), reaching 86.77 against our standing 80.98 —
> and it is null or negative on the other three tasks, because rel-event is the only one
> whose training period sits close enough to test for temporal proximity to mean anything
> (147-day span, 15-day gap; rel-trial's gap is 731 days). Recency is a one-dimensional
> retrieval key. If a *content* key does better on the tasks where time does nothing, that
> is this item, and it now has a measured baseline to beat rather than an argument.
> See `PERFORMANCE.md`, 2026-08-05.


**Bottleneck:** context length limits and memory scaling.
**Split this claim in two, because the halves have very different evidence.**

**The cost half is already demonstrated.** rel-avito matches full-context quality at
8.6% of its rows (10,000 of 116,598), which is the difference between requiring a 46 GB
GPU and running on CPU. `calibrate_context_size` ships this. Extending it toward
"millions of rows" is engineering with a known shape.

**The quality half — retrieval beating random subsampling — is unproven and the prior
evidence is discouraging.** `select_context(method="knn")` already exists here and was
never shown to help; an independent evaluation across four temporal relational databases
found entity-overlap retrieval disappointed on all but one (a pure forecasting task).

So: pursue the cost half as engineering. Treat the quality half as a hypothesis needing
a paired, multi-seed test at *equal context size* before any effort is spent scaling it.

## 3. Symbolic priors — enforcing domain constraints

**Bottleneck:** violations of domain logic; non-physical predictions.
**Why third:** genuinely novel, genuinely unmeasured, and not an AUC claim — a model
that never predicts a negative duration is better in a way ROC-AUC cannot express.

Least de-risked of the six. Needs a concrete constraint class chosen before anything can
be estimated: monotonicity in a named feature, bounded outputs, mutually exclusive
classes, and unit/dimensional consistency are all different problems with different
mechanisms (projection, constrained decoding, prior-side data generation).

Pick one constraint class and one dataset where violations are observable and countable
before scoping further.

## 4. Header fusion — column metadata as prior

**Bottleneck:** rich column metadata is discarded.
**Why below the line:** cheap and genuinely unexploited, but it is another *feature
signal*, which is the exact category that kept vanishing into the noise floor here.

The unexploited part is real: this package generates names like
`event_interest__count__30d` and then hands the model an anonymous matrix. Semantic
column embeddings are what CARTE and TabPFN-3's text handling exploit, and the
small-sample regime is where priors matter most.

Worth a bounded experiment, with the noise floor built into the test design from the
start: paired, multi-seed, and on the small-sample regime where the effect should be
largest. Do not run it as a single-seed A/B — that is how three claims in `DESIGN.md`
went wrong.

## 5. Tree-path hybrid — axis-aligned decision structure

**Bottleneck:** non-smooth axis-aligned boundaries; the residual gap to XGBoost.
**Why low:** the motivation is sound and well documented in the literature, but the fix
requires model surgery or pretraining. That is a different kind of project from anything
in this branch, which is deliberately featurisation-and-inference only and changes no
weights.

Revisit if pretraining capacity becomes available. Until then the cheaper probe is to
measure *where* TabICL loses to GBDT on these tasks — if the gap is not concentrated on
axis-aligned structure, the premise does not hold here.

## 6. Relational schemas — already largely done, with a documented ceiling

**Bottleneck:** flat single-table limitation.
**Status:** this is what the branch is. `flatten_relational`, `asof_statistics`, the
semiring/FAQ layer and the compiled WCOJ all exist and work.

Listed last not because it failed but because the informative work is finished and the
result is a *ceiling*, which should be stated before anyone invests further:

* calibrated scores sit 5–9 points behind RelGNN and TabPFN-REL on RelBench
* flattening provably cannot express self-referential many-to-many relations, cycles,
  diamonds, parallel edges, or multi-candidate-key schemas — the Smokers example is the
  textbook case, and no amount of feature engineering fixes it
* within flattening, which relations you traverse dominated everything computed over
  them, and even that was mostly inside the noise floor

"Extends TFMs to enterprise databases" is true with an asterisk. The remaining upside is
in schema *expressivity*, not in more aggregates. Concretely:

### 6a. Set-valued columns — worth doing, but does not lift the ceiling

The natural instinct for many-to-many is to store an array per row. It splits into two
problems and arrays only solve one.

*Representation* is the solvable half, and it collapses back to aggregation. TabICL needs
a fixed-width numeric matrix, so an array must be reduced anyway. Padding to length K
spends columns teaching the model that `[a,b,c]` and `[c,b,a]` are the same. Sorting
first makes it permutation-invariant but then it *is* an order statistic. The principled
form is a fixed-width **set encoding** -- quantiles, top-K histograms (already built),
VLAD residuals -- which is strictly better than mean/std but is not a new capability.

*The i.i.d. violation* is the half arrays cannot touch, and it is where the ceiling
actually is. When the signal is in neighbours' **labels** rather than their features,
storing their ids changes nothing: at fit time using those labels is leakage or
transduction, at predict time they are unknown.

The practical split: many-to-many where the neighbour's *features* carry the signal
(customers-products via transactions) is already handled. Where the neighbour's *label*
carries it, see 6b.

Worth building anyway: **quantile aggregates** (cheap, permutation-invariant, more than
mean/std) and **recency-ordered fixed windows** for temporal many-to-many -- "last 10
events" is the one case where a literal array is right, because the order is principled
rather than arbitrary.

### 6f. Text columns — RESOLVED MYSTERY, and probably the largest thing we ignore

**Their LightGBM baseline embeds text. Ours throws it away.** Two attempts to reproduce
RelBench's entity-only baseline landed 31 points short on rel-f1 (42.49 against 73.92) and
10.5 short on rel-trial. Reading `examples/lightgbm_entity.py` explains it: the baseline
merges the task table with the entity table exactly as we do, but then builds a
`torch_frame.data.Dataset` with `col_to_stype` from `get_stype_proposal` **and a
`TextEmbedderConfig`**. Free-text columns become embeddings.

Our pipeline drops them. `_numeric()` factorises non-numeric columns into arbitrary
integers, and `eval_entity_baseline` explicitly excludes near-unique object columns as
"identifiers" — which is right if the alternative is factorising them into row ids, and
badly wrong if the alternative is embedding them.

**Why this is probably the biggest single gap left.** rel-trial is our worst task (−7.07)
and its studies carry descriptions, eligibility criteria and intervention text; TabPFN-REL
reaches 76.43 there against our 69.36. The one lever we have never pulled is the one whose
content we discard entirely. It also reframes the DFS comparison in 6e: DFS aggregates
columns, and neither it nor we do anything with text.

**Do this before any further feature family.** Concretely: identify text columns via
`get_stype_proposal`, embed them with the same sentence encoder RelBench uses, and append
to the feature block — one variable, gated by standalone AUC per column as usual. The
`eval_entity_baseline` harness already isolates model from features and can measure it.

*Corollary:* the "LightGBM entity-only beats us" framing in earlier entries was never
apples-to-apples. It is not "raw columns beat your relational pipeline" — it is "text
embeddings beat a pipeline that ignores text". That is a more useful statement and a more
actionable one.

### 6e. Benchmark against Deep Feature Synthesis — DONE: two wins, two ties, 5-23x faster

`STATUS.md` now positions the relational layer against DFS (Kanter & Veeramachaneni 2015 /
Featuretools) analytically: sufficient statistics that compose exactly rather than stacked
primitives, as-of cutoffs as a prefix scan, semirings, a compiled WCOJ for cyclic patterns,
an explicit column budget — and, the only one that has moved a number, use of *other rows'
labels*, which DFS deliberately never does.

**None of that is measured.** "A generic flattening pipeline in front of a stock TabICL" is
a self-description appearing throughout these documents, and it is doing real rhetorical
work — it is what makes trailing RelGNN sound acceptable. Running Featuretools over the
same four tasks under the same calibrated protocol would turn it into a fact. Two outcomes,
both worth having: this layer beats DFS and the engineering is justified, or it does not
and the honest framing becomes "DFS plus label features plus a foundation model", which is
still a result and a much cheaper thing to maintain.

Cheap to run and the highest information-per-hour item currently open.

### 6d. Shared-key track record — DONE, +5.45, and the first change to the headline table

**rel-trial 66.50 → 69.36 calibrated, chosen by validation in 5/5 replicates.** Outcome
history among rows sharing a foreign key — sponsor, condition, facility, intervention —
as of this row's cutoff. `key_target_history` in `_propagation.py`,
`eval_track_record.py`, both controls passed.

The A/B is +5.45 (sd 0.30, 5/5) over an identical base, and +4.59 over a counts-only arm
that isolates connectivity from outcome. sd 0.30 against a ±0.6 floor makes it the most
reproducible effect in the project.

**Why this worked where 6b did not**, since the two are the same idea through different
structure. 6b reached neighbours through a *graph* and had to beat a base already at ~83
with a feature worth 67.8. This reaches them through the *schema* and had to beat a base at
~64 with three near-independent keys each worth ~61. The lesson is not "graphs bad" — it is
that a label-derived feature earns its place only where the existing features are not
already capturing the same base rate by other means, and that ratio is checkable up front
with a standalone AUC before anything is built.

**Generalise it.** Nothing in `key_target_history` is rel-trial-specific: it needs a link
table, labels, timestamps and a horizon. rel-avito has user/ad keys, rel-event has
user/event keys. Whether it transfers is the obvious next measurement, and the standalone
gate makes it cheap to find out before building per-task plumbing.

### 6b. Label propagation as a feature — BUILT, CONTROLLED, and it does not help

**Status: measured on rel-event, no gain. −0.74 at `n_estimators=4` and −0.38 at 8, both
inside the noise floor of zero.** The features are real, causal and individually
predictive; they add nothing on top of the relational features already there. Implemented
in `_propagation.py`, measured by `eval_propagation.py`, both negative controls passed.

**The 74 AUC figure this item was promoted on was wrong in two ways, and both corrections
matter more than the negative result.**

*Most of it was degree, not labels.* `labelled_degree` alone — how many friends hold a
resolved label, no label content whatsoever — scores **73.24**. The positive-rate feature
scores **67.78**, i.e. *below* the structural feature it was supposed to improve on. What
looked like "neighbours' labels predict your label" was substantially "well-connected
users differ from isolated ones".

*The rest included labels that had not happened yet.* The original 74.18 used every
training label regardless of time. A neighbour's outcome is not knowable at the
neighbour's prediction time — it resolves over the following 7 days — so restricting to
labels actually resolved at the query's cutoff drops it to 67.78. **~6.4 AUC of the
original number was read from the future.** `label_horizon` now enforces this.

Conditional on degree the label content is genuine (within-stratum AUC 64.6–73.5, and the
permutation test clears its null by 6.4 sd), so this is not a leak — it is a real effect
that the pipeline already captures by other means.

**Method note worth keeping.** `permutation_control` is the wrong control for a graph
feature. It asks whether the permuted score returns to 0.5, which holds only when the
feature's whole content is labels; a neighbour rate also encodes degree, which survives
permutation and predicts this target at 73. Judged against chance, a sound feature reads
as a leak — it did here, and the fix was `permutation_test` against the empirical null,
not a loosened tolerance. Getting that wrong in the other direction would have discarded
the feature; getting it wrong in this direction would have shipped a fake number.

The direct attack on the i.i.d. limit: inject neighbours' labels as features. `A^k · y`
with `y` the label indicator is "how many positives are reachable in k hops", and the
semiring layer already computes it -- this would be its first application on real signal.

| semiring | feature |
|---|---|
| `SUM_PRODUCT` | walk counts -- positives weighted by path multiplicity |
| `BOOLEAN` | distinct reachability |
| `MIN_PLUS` | **hop distance to the nearest positive**, often stronger than any count and degree-robust by construction |

`O(k·E)` sparse matvec, tractable on rel-event's 30.4M edges.

Two constraints decide it, neither computational:

* **Leakage.** Exclude self, use only labels known before the row's cutoff, never touch
  test labels. This is the feature family where a mistake yields a spectacular fake
  result, so it needs the negative control in *7* below before any number is believed.
* **Small-world saturation.** In a social graph the k-hop reachable set explodes toward
  the whole component. By k=3 on 30M edges, `reachable_count` risks becoming
  "component size" for every row, and the positive *fraction* converges to the global
  base rate -- worse than useless, because it still looks informative. Use fraction, not
  count; expect k=1,2, maybe 3; and measure the saturation curve rather than assuming it.

**Gate before building:** measure label homophily/assortativity at k=1. If a node's label
is uncorrelated with its neighbours', nothing downstream can help. One cheap number.

### 6c. Label-typed triangles — the degree-robust version of 6b

Raw triangle counts are largely determined by the degree sequence (see the degree-confound
section in `DESIGN.md`; a k-star count is exactly `C(d,k)`). But *whose labels close the
triangle* is not determined by degree, and it measures community cohesion rather than
connectivity: if your positive neighbours are also neighbours of each other, you sit
inside a positive-labelled community rather than merely adjacent to some positives.

This is computable with the **existing** `typed_triangle_counts`, by typing each edge with
the label pair of its endpoints and making "unknown at cutoff" a first-class type:

    P = known positive before cutoff,  N = known negative,  U = unknown (includes the ego)

That gives `{PP, PN, PU, NN, NU, UU}` -- exactly `MAX_EDGE_TYPES`. For an ego `v` (type
`U`) whose two mutually-connected friends are both known positive, the triple is
`(PP, PU, PU)`; both known negative gives `(NN, NU, NU)`; the mixed `(PN, PU, NU)` is the
bridging triangle *between* subgroups, its own signal. The ratio between the first two is
the "which community am I more embedded in" feature.

**The leakage rule is enforced by the typing itself** -- a node whose label is not known
before the cutoff is `U` by construction -- which is exactly the property this family
needs.

Costs: `k**3` = 216 joins at six types, and 56 sorted-triple columns, which is a lot of
features on tasks where feature count itself has been shown to hurt. Note also that the
measured gain from typing was only +0.021, but that was typing by *relation* type; typing
by *label* encodes homophily directly and is a different proposition.

**Gate:** do same-label pairs close triangles at a higher rate than the graph's baseline
transitivity? One number, and it is precisely the effect this feature would exploit.

---

# Further directions

Generated after the six above, and two of these outrank most of that list.

## 7. Leakage negative-control harness — build this before anything in 6b/6c

Permute the labels and recompute the feature. Any feature that still predicts is leaking.
It is a standard negative control, it is cheap, and it is the only thing standing between
"label propagation works" and a spectacular fake result.

This project's own record argues for it: several conclusions here were wrong for reasons
that produced *confident, plausible numbers* rather than errors. Label-derived features
are the highest-risk family yet attempted, and the harness costs a fraction of what one
retracted result costs.

Should also assert the temporal rule directly: recompute with cutoffs shifted earlier and
confirm scores degrade rather than improve.

## 8. Graph neighbours as ICL context — MEASURED, real, and NOT SELECTABLE ⚠

**The effect is real and reproduces:** +3.10 (1 hop) / +3.37 (2 hops) at 8 paired seeds,
+4.80 unstratified and +4.94 stratified at 3 seeds, 7/8 seeds positive, over a random
context of equal size on rel-event. Homophily lift +0.29. It also makes the score ~3x
more reproducible — the graph arm spans 3.1 points where random spans 9.1 — and beats the
full-context baseline on a quarter of the rows.

**And the calibrated protocol rejects it anyway: 82.01 ± 4.94 over 5 replicates, graph
chosen 1 time in 5.** That is the blocker, and it is not a measurement problem to be
solved with more seeds. Validation rates graph selection 74.1–74.5 across every seed —
stable to ±0.15 — while test rates the same selection +4.80 *above* random. The splits
are not the explanation: the standalone neighbour-label signal is 73.89 on val and 74.18
on test, and their connectivity is near-identical.

Two hypotheses are already dead. **Class balance:** unstratified selection really does
build a 0.019–0.046-positive context against a 0.163 base rate, because ranking by reach
ranks by popularity and hubs are negative — but stratifying it changes the test gap by
0.14, inside the floor. **Grid artefact:** the size-10000 cell was never a graph
configuration (pool is 7,111), and it is the only cell validation ever picked; that is now
dropped from the grid.

So the open question is no longer "does graph context help" — it does — but **"what
validation signal would ever select it?"** Until that is answered the effect cannot enter
the headline table, because a result you cannot choose without seeing test is not a result
you can report. Worth trying: select on a held-out slice of *train* rather than the
official val split; or select on the paired gap rather than the absolute score, since
pairing is what takes the noise from 2.28 to 0.29.

This remains the only idea here that attacks the i.i.d. limitation rather than adding a
feature — which is why it is worth the extra work rather than abandonment.

**The strongest idea here, and it bridges 2 and 6.**

TabICL attends from each test row to labelled context rows. If the context contains that
row's **graph neighbours with their labels**, then attention *is* one round of learned
message passing — structurally what a GNN layer does, obtained with no weight changes and
no retraining.

This reframes the retrieval question productively. Retrieval by *feature* similarity
disappointed both here and in an independent four-dataset evaluation. Retrieval by *graph
proximity* is a different hypothesis and directly targets the i.i.d. limitation that
feature-similarity retrieval never addressed.

It also composes with the context-size result: rel-avito matches full-context quality on
8.6% of its rows, so there is budget to spend on *which* rows without paying more.

Test: at equal context size, ego-graph neighbours versus random versus k-NN. Paired,
multi-seed. If neighbour-context beats random on a task with label homophily and not on
one without, the mechanism is confirmed rather than merely observed.

## 8b. Rel-LLM — prior work that solves item 8's blocker, and rebuts our framing

*"Large Language Models are Good Relational Learners"*, Fang Wu, Vijay Prakash Dwivedi,
Jure Leskovec. [arXiv 2506.05725](https://arxiv.org/abs/2506.05725),
[smiles724/Rel-LLM](https://github.com/smiles724/Rel-LLM). Leskovec's group — the same
group behind RelBench, so the protocol is theirs and the comparison is direct.

**Their pipeline:** build a heterogeneous entity graph from the tables → sample a
**temporal-aware subgraph at each prediction time** → encode with a GNN (GraphSAGE) →
project the embeddings into the LLM's space and serialise as JSON prompts → decode with a
**frozen** Llama-3.1, optionally with soft prompting. Framed as retrieval-augmented
generation over a database. Reported: all 7 RelBench datasets and 30 tasks, ~+2 AUROC over
RDL and over an "ICL+MLP" baseline, and a zero-shot "Rel-Zero" variant.

Base LLM is **Llama-3.2-1B** — small, not a frontier model — and scaling to 3B gives only
modest gains.

### Their Table 1, restricted to our four tasks (test AUROC)

| task | LightGBM | RDL | ICL | ICL+MLP | Rel-Zero | Rel-LLM | **ours** |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 73.92 | 75.54 | **88.47** | 87.36 | 70.64 | 82.22 | 80.70 |
| rel-event / user-ignore | 79.93 | 81.62 | 78.55 | **84.02** | 61.32 | 83.74 | 78.11 |
| rel-avito / user-visits | 53.05 | 66.20 | 60.28 | 64.98 | 56.17 | **67.01** | 64.85 |
| rel-trial / study-outcome | **70.09** | 68.60 | 55.72 | 68.38 | 59.02 | 71.04 | 69.36 |
| *average, all 11 tasks* | 63.66 | 75.83 | 69.63 | 76.83 | 63.42 | **77.82** | — |

**Five things worth acting on.**

**1. LightGBM on the entity table alone scores 70.09 on rel-trial. We score 69.36.** Their
LightGBM baseline uses *only the single entity table* — no relational features at all. Our
entire relational pipeline, plus the track record that was today's best result, does not
beat it. Worse, our A/B `base` arm was 63.82, so on this task the flattening machinery may
be *costing* us against a plain GBDT on raw columns. This is the baseline `RESEARCH 6e`
asks for, already published, and it should be reproduced locally before any further work
on rel-trial. It is the most important number in this file.

**2. ICL scores 88.47 on rel-f1 — the best result in their table, beating Rel-LLM's own
82.22.** ICL (Wydmuch et al. 2024) is in-context learning over serialised tabular rows,
which is the closest published method to what we do, and it beats us there by 7.8. On
rel-trial the same method collapses to 55.72. That spread is the finding: in-context
methods on relational data are wildly task-dependent, which is our own experience stated
by someone else.

**3. Their comparison column is not ours.** We quote TabPFN-REL / RelGNN / RDBLearn from
the TabPFN-3 report's Table 14; they quote LightGBM / RDL / ICL from RelBench. Both are
legitimate and they disagree substantially — on rel-trial we cite TabPFN-REL at 76.43,
while nothing in their table exceeds 71.04. `PERFORMANCE.md` should say which source each
comparison number comes from, because "state of the art" differs by source.

**4. rel-event's validation/test gap is a property of the task, not our bug.** RDL goes
**91.70 val → 81.62 test**; LightGBM 87.96 → 79.93. We spent real effort on rel-event's
val/test disagreement treating it as something we had introduced. Everyone has it.

**5. ICL is unstable across document-generation settings** (their Figure 2), which
independently corroborates our ±0.6 floor work and the 9.1-point spread we measured on
random context selection. Their answer is that a trained GNN encoder is more robust than
prompt construction — a direct argument against the retrieval-flavoured directions here.

*Honest note on the README:* it advertises "Rel-Zero performs competitively without
labels". The paper's own number is 63.42 average against LightGBM's 63.66 and RDL's 75.83.
The paper states this plainly; the README oversells it.

**Three things this is worth to us, in descending order.**

**1. It solves item 8's blocker, and we should copy the mechanism rather than the model.**
Our graph-neighbour context died on a static `user_friends` graph and an unselectable
validation signal; `PERFORMANCE.md` records the static-graph caveat as an upper bound we
never removed. *Temporal-aware subgraph sampling at each prediction time* is exactly the
missing piece, and it is item 9 already. That it works for them is evidence the idea is
sound and our implementation was the problem.

**2. Their "ICL+MLP" baseline is the closest published thing to what we do**, and they
beat it by ~2. Extracting that baseline's construction and per-task numbers is the
cheapest available check on whether our pipeline is competitive or merely plausible —
more informative than another lever on our own four tasks.

**3. It undercuts a claim these documents lean on.** We describe ourselves as "a generic
flattening pipeline in front of a stock TabICL, against systems built for relational
data", which frames trailing RelGNN as acceptable. Rel-LLM is a *foundation model* system
that keeps the GNN and beats the GNN baselines. So the honest contrast is not
"foundation model vs relational system" but **"how do you give a foundation model the
structure — a learned GNN encoder, or flattened columns?"** They chose the encoder. We
chose columns, which needs no training at all; that is our actual differentiator and it is
narrower and more interesting than the framing we have been using.

**What not to copy.** They need a Llama-3.1 and a trained GNN encoder per dataset. We
train nothing. If we land within a few points with zero training, that is a result worth
stating precisely — and it only means anything against their numbers, not ours.

**Also a rebuke on coverage.** They report 30 tasks; we report 4. Several conclusions here
are single-task, and rel-trial's +5.45 versus rel-event's +1.33-over-counts already shows
how badly one task generalises to another.

## 9. Time-respecting propagation — the causally sound form of 6b

A path only carries influence if timestamps increase along it. "Positives reachable by a
**time-respecting** path" is a far stronger causal claim than static reachability, and
`temporal_motif_features` is the start of the machinery.

It also fixes a caveat already recorded in `DESIGN.md`: rel-event's `user_friends` carries
no timestamp, so the graph is static and any lift there is an upper bound. Time-respecting
paths make that limitation explicit rather than silent.

## 10. Context-resampling uncertainty — turn the measurement noise into a product

Absolute scores move with sd 2.28 across context subsamples on rel-event. That variance
has been a nuisance all along; it is also **information**. Resampling the context yields a
predictive distribution at no architectural cost, giving uncertainty estimates that are
honest about the dominant source of variance in this model — which rows happened to be in
context.

Composes directly with direction 1 (attribution): a prediction that is stable across
context resamples is one no single row controls, which is a usable robustness measure.
Cheap, since `n_estimators` already does something structurally similar.

## 11. Incremental maintenance — recompute only what changed

The semiring layer's `invertible` flag already distinguishes statistics that can be
maintained under deletion from those that cannot. That is the foundation for
factorised incremental view maintenance: on a daily refresh, update the aggregates the new
rows touch instead of rebuilding every feature from scratch.

A capability claim rather than an accuracy one, which is why it survives the noise floor.
It is also the difference between a batch experiment and something that runs in
production. `SUM_PRODUCT` statistics are maintainable; `MIN_PLUS`/`MAX_PLUS` extrema are
not, and needing a recompute for those is a known, bounded cost.

## 12. Model-side column selection

`max_columns` prunes by non-null coverage, which is target-free but blind -- and the same
setting is worth +3.0 on one task and -19.5 on another. The column-embedding stage
produces a per-column representation *inside the model*; its attention is a far better
relevance signal than coverage.

Prune using the model's own view of the columns, on a small context, then fit on the
survivors. Directly targets the one setting measured to have a 22-point spread, which
makes it the highest-leverage of the feature-side ideas even though it is still an
accuracy claim.

---

## Sequencing note

Nothing here should start before `TODO.md` item 1 closes. That question — whether a
setting's verdict flips with `n_estimators` — is currently capable of inverting any
comparison made on these tasks, which would silently corrupt the evaluation of any
direction above.

## Model selection under distribution shift — literature review (2026-08-10)

**Why this review and not another tabular-model one.** Six papers on tabular foundation
models produced one usable lead. But the binding constraint here is not the model: it is
that **eleven interventions were real on fixed arms and about zero after selection**, and
the diagnosis is precise — validation entities appear in train **81.2%** of the time against
test's **58.6%**, so validation overrates exactly the arms that lean on history. That is a
*model selection under covariate shift* problem, and this project had never read that
literature.

### What we had already tried, and the pattern in it

| attempt | what it changed | result |
|---|---|---|
| `--gap-validation` | more temporal separation val→test | **worse** (78.71 vs 80.25) |
| `--decide-fit-pool` | which rows are fitted on | failed |
| `--abstain` | veto if ranking fails on a later val half | refuted |
| `--ensemble-configs` | average top-k instead of argmax | refuted twice |
| `--match-novelty` | resample val to test's seen/unseen mix | **widened** the gap |
| validation-noise fallback | fall back to `base` inside the noise band | 2 of 6, worse than chance |
| `--drop-stale-arms` | remove arms whose features degrade | **+0.65 / +0.49, only survivor** |

**Four of the six failures tried to read a better answer out of the same validation scores.**
The two that changed something structural — who is scored, what is eligible — are the only
ones that produced anything, and one of them works.

### 1. Random validation splits beat temporal ones — and we assumed the opposite

**"Understanding the Limits of Deep Tabular Methods with Temporal Shift"**
([arXiv 2502.20260](https://arxiv.org/abs/2502.20260)): *"While existing approaches use
temporal ordering for splitting validation set, they show that **even a random split can
significantly improve model performance**."*

**This project went the other way.** `--gap-validation` widened the val→test time gap and
measured **worse** (78.71 against 80.25); the entry concluded the instrument was faulty. The
literature says the sign was wrong: the fix is *less* temporal separation, not more. **The
opposite direction was never tested.**

It is also consistent with our own measurement that selection is noise-dominated — margin
**0.35** against validation noise **0.69**, with margin < noise in **35 of 41 blocks**. A
random split yields a larger, lower-variance validation signal; trading a little bias for a
lot of variance is exactly the trade a noise-dominated selector wants.

**Cheapest experiment available, and directly contradicts a standing conclusion.**

### 2. HyperTime — lexicographic (average, worst-case) over chronological folds

**"HyperTime: Hyperparameter Optimization for Combating Temporal Distribution Shifts"**
([arXiv 2305.18421](https://arxiv.org/abs/2305.18421)). Split validation into chronological
folds; compute `L_avg` and `L_worst`; select **lexicographically with a tolerance band** —
keep configs within κ of the best average, then among those take the best worst-case.

**Fits our measured situation unusually well.** Because margin < noise in 35 of 41 blocks,
there are many configs statistically tied on average validation. HyperTime breaks those ties
by *temporal robustness* rather than by noise.

**Distinct from `--abstain`, which we refuted.** That used the later validation half as a
**veto**; this uses folds as a **tiebreak among near-optimal configs**. `split_val` already
computes early/late halves per configuration, so most of the machinery exists.

### 3. Rolling window + tournament

**"Model Assessment and Selection under Temporal Distribution Shift"**
([arXiv 2402.08672](https://arxiv.org/abs/2402.08672), ICML 2024). An adaptive rolling
window over current *and historical* epochs to estimate generalisation error, with pairwise
comparisons folded into a single-elimination tournament. Theoretically grounded and heavier
than the above; the rolling window needs multiple historical validation epochs, which our
train span could supply.

### 4. Importance-weighted cross-validation — the classical answer, and we already failed a crude version

**Sugiyama et al., "Covariate Shift Adaptation by Importance Weighted Cross Validation"**
(JMLR 2007). Weight each validation row by the density ratio `p_test(x) / p_val(x)`; the
weighted risk is an almost unbiased estimator of the target risk. **Unbiased but with
unbounded variance**, and the density ratio must itself be estimated — hard in our
130–360-column frames.

**`--match-novelty` was a one-variable version of this** (matching only the seen/unseen
entity mix) and it *widened* the gap. That is weak evidence against the family, not strong:
a single binary covariate is a poor stand-in for a density ratio. But given the variance
warning and our already noise-dominated selector, this is the least attractive of the four.

### 5. A drift-aware backbone, which joins this thread to the TabFM one

**"Drift-Resilient TabPFN"** ([arXiv 2411.10634](https://arxiv.org/abs/2411.10634), NeurIPS
2024, code at `automl/Drift-Resilient_TabPFN`). A PFN pretrained on synthetic datasets from
**evolving structural causal models**, so non-stationarity is in the prior. Reports accuracy
0.688 → **0.744** and ROC AUC 0.786 → **0.832** against the strongest baselines across 18
synthetic and real datasets, beating XGB, CatBoost, TabPFN and the Wild-Time methods. Frozen,
no hyperparameter tuning, "small to moderately sized datasets" — and our context is 10,000
rows, which may sit inside that.

**Our setting is exactly what it was built for**: train → validation → test in time order
with measured drift.

### What to run, in order of expected value per pod-hour

1. ~~**Random validation split.**~~ **RUN 2026-08-10 — REFUTED.** +0.02 (SE 0.04) on
   user-clicks, −0.12 on rel-trial, −0.21 on rel-event, 12 replicates each. It changed the
   selected configuration on **5 to 8 of every 12 seeds** and moved test by nothing, so this
   is a refutation of an active intervention, not a null from one that never fired.
   **The mechanism was wrong on inspection, and that was visible beforehand:** a random slice
   of TRAIN has *higher* entity overlap with the remaining training rows than a temporal
   slice does, and the diagnosed failure here is that validation overrates history-leaning
   arms *because* its entities appear in train 81.2% of the time against test's 58.6%. A
   random split makes that mismatch worse. arXiv 2502.20260 argues from variance, not
   population — importing it was importing a fix for a different problem.
2. ~~**HyperTime lexicographic selection.**~~ **RUN 2026-08-10 — NOT ADOPTABLE.** −0.13 on
   rel-trial (fires 8/12, does nothing), **−0.73 on rel-event** (fires 7/12), +0.02 on
   user-clicks where it fires **1/12** and is therefore uninformative rather than tested.
   No gain anywhere and a suggestive loss on a control; the loss shrinks to −0.44, so the
   defensible claim is "no gain, with a possible cost", not "it costs 0.73".
3. **Drift-Resilient TabPFN as a backbone**, alongside TabFM. **Not measured — the
   integration is the obstacle, and here is the map so nobody rediscovers it.** Three
   attempts on 2026-08-10, each failing differently and each narrowing it:

   | install | outcome |
   |---|---|
   | plain `pip install git+…` | resolves **numpy 2.x** under a torch built against 1.x. Torch then imports with "Failed to initialize NumPy: _ARRAY_API not found" — a *warning* — so the gate passed and the round ran with every tensor↔array conversion dead |
   | `numpy<2` pinned | bridge fine, but pip dragged **torch 2.4.1 → 2.1.2**, below the 2.2 this model needs (`Tensor.all(dim=tuple)` raises beneath it, inside the forward pass) |
   | `--no-deps` | torch and numpy untouched; import dies at `NameError: KDITransformer is not defined` — the fork uses it at class-definition time and its dependency was excluded |
   | `--no-deps` **+ `kditransform`** | **same `NameError`.** Adding the package under
     `--no-deps` does not satisfy it — either the flag also starved `kditransform` itself, or
     the fork expects a different provider of that symbol |

   **CLOSED AT FOUR ATTEMPTS, deliberately.** This is a *screen*: `eval_backbone` is a
   fixed-configuration test-side comparison, and its own footer says a gain there is not
   table-eligible until the calibrated protocol picks it — the step that has killed nine real
   effects on this benchmark. Four pod cycles on dependency resolution for a result that could
   not be reported anyway is already past the point where it earns its keep. The next person
   should start by resolving `KDITransformer` in a scratch environment, not on a pod.

   Two guards were added because of this and are worth keeping
   regardless of whether the backbone is ever adopted: `_pod_setup.sh` now exercises the
   numpy↔torch bridge in both directions *and* re-checks the torch floor after any extra
   install, because "the package installed" and "the environment still works" turned out to
   be two different questions, twice.

   The class is `tabpfn.TabPFNDistShiftClassifier` — learned only because the arm was written
   to print the module's real exports on failure instead of guessing. All four of the names
   guessed up front were wrong.
4. **IWCV** last, and only if 1–3 fail: highest variance, and the crude version already lost.

### What 1 and 2 together establish, which neither shows alone

On user-clicks the pick changed on 7 of 12 seeds and test moved 0.02. So **the candidates the
selector chooses among are near-equivalent on test** — yet tuning costs 1.29 there. Both are
only true if the tuned candidates are collectively worse than the untuned `base` arm and
shuffling among them is irrelevant. Every instrument this project has built reshuffles the
pick, which is why nine of them have failed: **they attack the wrong quantity.** The lever is
not which configuration validation picks; it is whether to select at all, or to change what
quantity is being estimated — which is directions 5 and 6 below.

### 8. The benchmark is far larger than the twelve tasks we run (found 2026-08-11)

Incidental to the rel-arxiv screen, and more consequential than the screen was. The pod
printed the installed registry before asking it for anything, and it lists **33 datasets**
where this project knows seven, and far more tasks per dataset than the one or two we run:

| dataset | tasks it registers |
|---|---|
| rel-stack | `user-badge`, `user-engagement`, **`badges-class`**, **`post-votes`**, **`post-post-related`**, **`user-post-comment`** |
| rel-trial | `study-outcome`, plus **`site-success`**, **`study-adverse`**, **`studies-has_dmc`**, `eligibilities-adult/child`, `studies-enrollment`, `condition-sponsor-run`, `site-sponsor-run` |
| rel-f1 | `driver-dnf`, `driver-top3`, plus **`driver-position`**, **`qualifying-position`**, `results-position`, `driver-circuit-compete` |
| rel-avito | `user-clicks`, `user-visits`, plus **`ad-ctr`**, **`searchinfo-isuserloggedon`**, `searchstream-click`, `user-ad-visit` |
| rel-amazon | `user-churn`, `item-churn`, plus `item-ltv`, `user-ltv`, `review-rating`, `user-item-*` |
| rel-arxiv | `paper-citation`, **`author-category`**, `author-publication`, `paper-paper-cocitation` |
| rel-event | `user-ignore`, `user-repeat`, plus `user-attendance`, `users-birthyear`, `event_interest-*` |

And whole families we had never seen: **`rel-ratebeer`, `rel-salt`, `rel-mimic`**, eight
`dbinfer-*` datasets, and the TGB temporal-graph benchmarks (`tgbl-*`, `tgbn-*`, `thgl-*`),
whose `src-dst-mrr` tasks are link prediction on genuinely temporal graphs.

**Why this matters more than a longer list.** Three standing conclusions are qualified by it:

* **The twelve-task table is a slice, not the benchmark.** Its worth is that it reproduces the
  published `Avg AUROC` to the digit for all nine comparison methods, and that stays true. But
  "our position on RelBench" means our position on the twelve tasks that report chose.
* **Several bolded tasks are binary classification on datasets already downloaded**, so they
  cost a round rather than a port. `study-adverse`, `site-success`, `ad-ctr` and
  `badges-class` are the obvious candidates, and they would widen the worst-case-regret
  evidence base that currently rests on rel-trial's single cell.
* **`author-category` exists**, which is the label the motif screen actually wanted -- co-author
  community predicting research area, rather than paper-citation's popularity. It is multiclass,
  which this harness does not support.

**And a caution about how this was found.** We had run rounds against this registry for weeks
without ever printing it, and the version banner still reports `relbench (no __version__)`, so
no round on record can say which release produced its numbers. Listing a registry costs
seconds; that is a poor reason to have gone this long without one.

### 7. Conditional tuning — REFUTED 2026-08-11 on the first held-out task

**The prediction below was falsified by rel-hm/user-churn: tuning gain −0.76 (SE 0.16,
t = −4.90, 7 of 8 seeds negative, shrunk −0.69).** The criterion reports coverage HOLDS there
(0.96–1.03), so it says *tune*, and tuning loses three quarters of a point — past the −0.6
falsifier fixed in advance, on both the raw and the shrunk figure.

So the shared-key coverage ratio **does not identify where tuning is safe**. It flagged
rel-avito and rel-f1 and looked convincing on the seven tasks that produced the gains; the
first task it had never seen breaks it. That is the whole reason the test used tasks which
informed neither the criterion nor the table, and the reason the prediction was written down
before any of them ran.

**What survives:** nothing of the conditional policy. The decision reverts to always-tune (A)
versus fixed defaults (B) on worst-case regret. Adding rel-hm as an eighth task leaves that
verdict unchanged — always-tune's worst case is still 0.79 (user-clicks, now joined by
rel-hm's 0.76) against never-tune's 2.17 (rel-trial) — but it makes the picture worse in a way
the verdict hides: **three of eight tasks now lose from tuning**, and the case for A rests
entirely on rel-trial continuing to be worth +2.17.

The remaining held-out tasks were still worth running, for a different reason than they were
started: not to rescue the criterion, which is dead, but because a task where tuning loses
more than 2.17 would flip the A/B verdict outright.

#### The held-out set, COMPLETE (2026-08-12)

8 replicates each, paired by seed, tuned = the calibrated protocol and base = the fixed `base`
arm of the same task on the same pod. All four ran at `--max-columns 2 --offload cpu
--train-pool 300000` on an H100 SXM, the tier that made the three largest tasks fit at all.

| task | tuned | base | gain | SE | t | seeds + | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| rel-hm/user-churn | — | — | **−0.76** | 0.16 | −4.90 | 1/8 | **refutes** |
| rel-amazon/user-churn | 67.11 | 66.96 | +0.15 | 0.13 | 1.14 | 5/8 | null |
| rel-stack/user-engagement | 89.70 | 89.57 | +0.12 | 0.14 | 0.90 | 5/8 | null |
| rel-stack/user-badge | 83.95 | 83.81 | +0.14 | 0.18 | 0.79 | 5/8 | null |
| rel-amazon/item-churn | 80.16 | 80.12 | +0.04 | 0.08 | 0.48 | 4/8 | null |

**The four nulls are the outcome the pre-registration warned would confirm little**, and two
of them are the very tasks named there: item-churn and user-badge have most arms excluded by
their own leak controls, so selection has almost nothing to choose between. user-engagement was
not on that list and still came in at +0.12 — inside the floor, five seeds of eight positive.

**Nothing here rescues the criterion, and the set is now complete without flipping A/B.** The
criterion was already dead on rel-hm; four nulls neither revive nor further damage it. For the
A/B decision what mattered was whether any held-out task loses more than rel-trial's +2.17, and
the worst of the five is rel-hm at −0.76 — which merely ties user-clicks rather than beating it.
**Always-tune's worst case is unchanged at 0.79, never-tune's at 2.17, and there are no cells
left to measure.** The decision is now made on a complete twelve-task picture rather than a
seven-task one.

**user-badge separates the two axes selection sweeps, which no other cell has.** Validation
chose `base` on **all eight seeds** — so the arm axis contributed exactly nothing, as the leak
controls predicted, and the entire +0.14 came from the *context size*: 2,500 rows on four
seeds, 5,000 on two, 10,000 on two, against the fixed arm's 10,000. A smaller context beat a
larger one and validation could see it. That is worth holding next to the recency thread, where
validation could *not* see a small context on rel-event — the difference is that rel-event's
win needs the small context to also be *recent*, and here plain size was enough.

**And the number it replaces was honest.** The standing table carried 83.80 from four salvaged
replicates; eight clean ones put the fixed arm at 83.81 and the calibrated protocol at 83.95.
Six earlier attempts on this task died `rc=137` during aggregation; what cleared it was an
H200's memory tier, not any flag.

**A side result worth keeping:** rel-stack/user-engagement now has a clean 8-replicate
**89.70 ± 0.18** (base arm 89.57), replacing the salvaged 89.33 that STATUS carried from a
partial run.

#### The original pre-registration, kept verbatim

The re-measured table says tuning is a rel-trial-specific win (+2.17), a user-clicks-specific
loss (−0.79), and nothing measurable on the other five. A conditional policy — tune only where
it pays — beats both always-tune (worst-case regret 0.79) and never-tune (2.17) *on that
table*. But choosing the condition after seeing which tasks won is fitting the rule to the
answer, which is the objection that keeps `--drop-stale-arms` out of the standing table. So
the condition is taken from a criterion that ALREADY EXISTED, and tested on tasks that did not
inform it.

**The criterion, unchanged from `coverage_probe.py` (commit 2039c50, before this table):** the
shared-key block's coverage ratio between validation and test, with the threshold
`--drop-stale-arms` already used, 0.8. It is label-free — links, timestamps and the existence
of a label column, never its values — and computable before any test label is read.

**What it says about the seven tasks it was not chosen on.** Coverage collapses on exactly
five RelBenchV1 tasks, all on rel-avito and rel-f1. Lining that up against the re-measured
gains, without touching either:

| coverage | tasks | re-measured gains |
|---|---|---|
| collapses (<0.8) | rel-avito ×2, rel-f1 ×2 | −0.79, −0.36, +0.52, +0.01 |
| holds (0.96–1.03) | rel-trial, rel-event ×2 | **+2.17**, +0.27, +0.10 |

"Tune only where coverage holds" would have worst-case regret **0.52** (forgoing driver-dnf),
against 0.79 always-tune and 2.17 never-tune. Encouraging — and NOT yet evidence, because
these are the same seven tasks that produced the gains.

**THE PRE-REGISTERED TEST.** `coverage_probe` reports all five held-out RelBenchV1 tasks at
0.96–1.03: coverage HOLDS on every one. So the rule says **tune on all five**, and therefore
predicts:

> **No held-out task shows tuning losing materially** (nothing below −0.6, the floor).

**What refutes it:** any of rel-hm/user-churn, rel-stack/user-engagement,
rel-stack/user-badge, rel-amazon/user-churn or rel-amazon/item-churn losing more than 0.6 by
tuning. One such task and the criterion does not identify where tuning is safe.

**Stated in advance, the test is weaker than it looks.** Two of the five (rel-stack/user-badge,
rel-amazon/item-churn) have five of seven arms excluded by their own leak controls, so
selection has almost nothing to choose between and a gain near zero there confirms little. A
*loss* would still refute. Reporting this now so a null is not later read as support.

### 6. Temporal-distance extrapolation — BUILT AND REFUTED, 2026-08-10 (`--select-extrapolate`)

**Verdict first: it cannot pay for itself, and where it runs it is harmful.** Three tasks,
three different reasons, none of them fixable with more replicates:

| task | delta | what happened |
|---|---:|---|
| rel-avito/user-clicks | — | **cannot run.** 2/3/4-day gaps select an identical pool; two points, no verifiable slope |
| rel-trial | **+0.03** (SE 0.18) | target inside the bracket → reduces to gap-matched selection → does nothing. But its **precondition costs −0.58**: the pools only hold 6,483 rows, so the context grid must be capped, and that cap is what the naive −0.55 was measuring |
| rel-event | **−1.84** (SE 1.03, 7/8 seeds worse) | the only genuine case, and it *hurts*; variance nearly doubles and one seed loses 8.38 |

**The mechanism is the useful part.** rel-event's eight fitted slopes were `+0.031 +0.046
+0.046 +0.116 / −0.002 −0.020 −0.024 −0.042` — **four positive, four negative.** The sign is a
coin flip across seeds, so there is no stable decay to estimate; each 3-point fit looks locally
tidy enough to survive shrinkage, and the rule then changed the pick on 6 of 8 seeds. It
injects noise into selection rather than correcting a bias.

**And the precondition is structural.** Stale pools must be large enough to serve the context
grid, or `draw` clamps and the distance axis silently becomes a pool-size axis. Capping the
grid is therefore not optional — and on rel-trial that cap costs 0.58 against a rule worth
0.03. The machinery cannot pay for itself even where the idea is sound.

**What bounds applicability is temporal RESOLUTION, not span.** 36,741 rel-avito rows precede
a 2-day cutoff and the same 36,741 precede a 4-day one: the label stream has holes wider than
the distinctions the method needs. "Use a longer history" does not fix that.

Kept in the tree, off by default, with its guards and 15 tests — the negative result is worth
more than the code, and the guards (refuse under three distinct gaps, drop duplicate pools,
discard an unverifiable two-point slope) are reusable.

---

#### Original design notes, retained for the reasoning

The only remaining idea that changes *what is estimated* rather than how the same ranking is
read. Score each candidate at several train→query distances inside train, fit the slope, and
evaluate the line where test actually sits. Two properties earned from the failures above:

* **It subsumes `--gap-validation` and fixes its confound.** That was this rule with one gap,
  no slope, and a pool whose size shrank as the gap grew, so its "distance" effect was partly
  a pool-size effect — and pool size is not a quantity that differs between validation and
  test. Here every pool is the last *N* rows before its cutoff: same size, only staleness moves.
* **It refuses rather than degrades.** Fewer than two usable gaps raises, instead of quietly
  becoming ordinary validation at an odd cutoff — the failure mode `--select-lexi` shipped with.

Aimed squarely at the measured shape of the problem: rel-event's context-recency effect is
worth **+7.50** on test at a 1,000-row context and validation scores it *low*, because
validation sits at a nearer period where more context helps and a small recent context does
not. An effect whose value grows with distance from the training period is exactly what a
slope can see and a single near-period ranking cannot.

#### Measured feasibility, 2026-08-10 — and it is the binding constraint, not the idea

The rule needs pools that are stale by the requested gap AND large enough to serve the
context grid. Both benchmarks tested so far fail to give it room, for *different* structural
reasons, and neither failure is fixable with more replicates:

| task | train→test gap | what the data supports | consequence |
|---|---:|---|---|
| rel-trial | 731 d | all of 366/731/1096 d, pools 6,483 | target sits **inside** the bracket, so the slope term is zero by construction → this is **gap-matched selection**, not extrapolation |
| rel-avito/user-clicks | 10 d | 2/3/4 d select an **identical pool** (no rows exist between them); only 5 d differs | **two** distinct points → a line with no residual → **refused** |

**rel-avito's timestamps are the obstacle.** 36,741 rows precede a 2-day cutoff and the same
36,741 precede a 4-day one: the label stream has holes wider than the gaps we need to
distinguish. Temporal resolution, not span, is what bounds this method — a distinction worth
keeping, because "use a longer history" does not fix it.

So on the task the whole selection thread is about, this direction **cannot be run at all**.
That is a stronger and cheaper answer than a null would have been, and it arrived from the
rule's own diagnostics rather than from a result that needed interpreting. What remains is
rel-event, whose 15-day gap against a 7-day validation gap is the widest ratio in the
benchmark and the only place a genuine slope may be measurable.
