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
