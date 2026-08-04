# Research directions

Six candidate directions, ordered by expected value rather than by expected benefit.
`TODO.md` is the open engineering work; this is what to build after it.

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
in schema *expressivity* (GFS covers all k-hop join paths rather than one traversal), not
in more aggregates.

---

## Sequencing note

Nothing here should start before `TODO.md` item 1 closes. That question — whether a
setting's verdict flips with `n_estimators` — is currently capable of inverting any
comparison made on these tasks, which would silently corrupt the evaluation of any
direction above.
