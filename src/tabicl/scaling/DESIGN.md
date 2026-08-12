# Porting TabPFN-3 scaling techniques to TabICLv2

Design + scope for four upgrades derived from *TabPFN-3: Technical Report*
(arXiv 2605.13986, Prior Labs, 2026-05-12).

> This file is the **history log**: derivations, what was tried, what failed, what the
> measurements actually said, and the reasoning behind each decision. For the current
> state of the package — API, configuration, standing results, known limits — see
> [STATUS.md](STATUS.md).

## Why this is cheaper than it looks

TabPFN-3 **adopted TabICL's architecture**, not the reverse. Figure 5: *"Architecture
of TabPFN-3, adapted from the TabICLv2 architecture."* §2 states TabPFN-3 abandoned
TabPFN-2.x's alternating row/column attention and moved to Qu et al.'s two-stage
row-compression, including TabICLv2's cyclic-triplet feature grouping and
target-aware embeddings.

So these are not cross-architecture ports. Both models are now
`cell embed -> column embed (inducing points) -> row aggregate -> row-level ICL`.

Concretely, TabICLv2 already has the structures each technique needs:

| TabPFN-3 needs | TabICLv2 has |
|---|---|
| inducing-point column stage | `SetTransformer` / `InducedSelfAttentionBlock`, `num_inds=128` |
| cached inducing summary | `ISAB.forward_with_cache` stores K/V of `hidden` |
| ICL KV cache | `Encoder.forward_with_cache`, `kv_cache.py` |
| frozen row representation | output of `row_interaction`, fed to `icl_predictor` |

Both use **128 inducing points**. TabICLv2's ICL stack is 12 blocks x 8 heads x
`d_head=16`.

## 1. Row-chunked inference

**Problem.** `ColEmbedding` materializes an `(N, C, d)` activation. At
`N=200k, C=100, d=128` in fp32 that is 10.2 GB — an OOM on a 12 GB 3060.
TabICLv2's current answer is CPU/disk offload; TabPFN-3 §2.4.1 notes this costs
~250 GB host RAM at `1M x 500`, or ~4x slowdown.

**Key enabling observation.** In ISAB stage 2 the call is
`multihead_attn2(src, hidden, hidden, need_kv=True)`. The returned `k_proj`/`v_proj`
are projections of **`hidden` only** — they do not depend on `src`. So the cache can
be populated with a 1-row dummy query at negligible cost, then real rows streamed
through stage 2 against it.

**Scheme** (exactly equivalent to unchunked, per TabPFN-3's two-phase design):

- *Phase A* — run the ISAB stack over **training rows only** to obtain each block's
  `hidden`, and store its K/V. Block `i+1`'s `hidden` depends on block `i`'s output
  for training rows, so phase A must run the full stack; it is bounded by
  `n_train x C x d`, and is itself chunked over the (independent) column axis.
- *Phase B* — stream **all** rows in fixed-size chunks; within a chunk run all
  blocks' stage 2 against the cached K/V. A row's stage-2 output depends only on
  that row and `hidden`, so chunks are independent.

Exactness holds because phase B recomputes precisely what the unchunked path would,
against identical `hidden`.

**Scope.** Wrapper over the public module API; no edits to `_model/layers.py`.
Not applied when `target_aware` mixed-radix ensembling is active (rare;
falls back to unchunked).

## 2. Reduced KV cache via multi-query attention

**Scoping finding — this one is not free.** TabPFN-3 *trained* with a single KV head
for test->train cross-attention. TabICLv2's released checkpoint has 8 full KV heads.
Converting post hoc means collapsing 8 heads into 1, which is an approximation, not a
refactor. **A faithful port requires pretraining.**

What is deliverable without pretraining:

- an MQA-capable cache path (correct, and training-ready);
- a post-hoc `mean` / `first` head-collapse so the size win is measurable today;
- honest measurement of the resulting accuracy degradation.

Expected size win is exactly `nhead = 8x`, matching the paper's "factor of eight".
ICL cache is `12 blocks x 2 x N x 8 x 16 x 4 B` = 12.3 KB/row -> 1.5 KB/row.

## 3. Relational data via table flattening

TabPFN-3 does **not** add relational schemas to its prior. §3.4: TabPFN-REL follows
RDBLearn, which converts tabular foundation models into relational ones by
*"automatically flattening the underlying database into a table"*. So this is a
featurization wrapper — no model change, no retraining.

**Scope.** Depth-k aggregation over foreign keys (count/mean/sum/min/max/std for
numerics, mode/nunique for categoricals), with timestamp truncation so no child row
after the cutoff leaks in. Pure pandas.

Multi-hop (`user -> order -> item`) is supported by giving a `Table` its own
`primary_key` and `children`; the deeper level folds in first and the entity's cutoff
propagates down, so a grandchild recorded after the prediction time is still excluded.

**Multi-hop uses factorized aggregation, and has to.** The naive approach --
aggregate items into orders, then aggregate those aggregates into users -- computes a
mean of means, which is not the mean. For a user with a 3-item order (prices 1, 2, 3)
and a 1-item order (price 100), it reports **51.0** where the true average item price
is **26.5**. The error is unbounded and grows with how unevenly children are
distributed.

The fix is the standard one from factorized databases (Olteanu & Schleich; FAQ /
semiring aggregation over a join tree): push only *decomposable* statistics up the
tree and derive the rest at the root.

| carried up each hop | combiner |
|---|---|
| `count`, `sum`, `sumsq` | `sum` |
| `min` | `min` |
| `max` | `max` |

Each is associative, so a chain of joins collapses bottom-up without ever
materialising the join. `mean` and `std` are *not* decomposable, but they are
functions of statistics that are -- `mean = sum/count`,
`var = sumsq/count - mean^2` -- so they are computed once at the root. `sumsq` is
then dropped; it is a carrier, not a feature.

Two consequences:

* **Correctness.** A test checks every statistic against aggregating the fully
  materialised join, and they agree to 1e-9.
* **Cost.** Statistics roll up 1:1 rather than cross-multiplying, so depth-k is
  linear. The earlier nested scheme turned one grandchild column into 25; this keeps
  it at ~7, and depth 3 adds no more.

### Look-back windows

`Table(windows=[...])` emits an extra block of statistics per window, over child rows
in `[cutoff - window, cutoff)`, alongside the all-history block. Recency is usually the
strongest thing a history carries and an all-time mean dilutes it -- a driver's
lifetime average says little about current form.

On rel-f1 / driver-top3:

| features | count | TabICL | GBDT |
|---|---:|---:|---:|
| all history only | 147 | 0.8125 | 0.7467 |
| + 365d | 288 | 0.8834 | 0.8376 |
| + 90d, 365d | 429 | 0.8984 | 0.8527 |
| + 30d, 90d, 365d | 570 | **0.9011** | 0.8642 |

**+0.089 AUC**, the largest feature-engineering gain measured here, from a filter
inside the existing cutoff logic rather than any new machinery. TabICL leads GBDT at
every level. Most of it arrives with the first window; the third adds +0.003, so the
count is a cost/benefit choice, not a free lunch -- 570 columns approaches TabICL's
comfortable feature range.

**It does not generalise.** Repeating the A/B across five RelBench tasks, routed
through `asof_statistics` so the ten-child schemas finish at all:

| dataset / task | all history | + 30d,365d | delta |
|---|---:|---:|---:|
| rel-f1 / driver-top3 | 0.7874 | 0.8874 | **+0.100** |
| rel-trial / study-outcome | 0.5010 | 0.5393 | +0.038 |
| rel-avito / user-visits | 0.6282 | 0.6288 | +0.001 |
| rel-event / user-ignore | 0.5485 | 0.5428 | -0.006 |
| rel-event / user-repeat | 0.5653 | 0.5228 | **-0.043** |

Median +0.001. The rel-f1 result is real and reproducible -- it appears again here on a
different feature set -- but it is a property of that task, not of windows.

The pattern in *when* it works is the useful part. rel-f1 is the only task whose
features work well at all (0.88); the rest sit between 0.50 and 0.63. Windows triple
the column count, so where signal is weak the extra columns are mostly noise, and the
worst loss is the task with a single sparse child table (13 -> 39 features, -0.043).
Recency plausibly matters more for driver form, which changes season to season, than
for event attendance.

Two qualifications on the weak rows: rel-event received only one child table because
`event_attendees` (8.4M rows) exceeded the harness size cap, so those arms are
under-featured rather than cleanly tested; and rel-trial's baseline is chance, so its
gain is movement off a floor.

Consistent with the motif finding that relation *choice* beat typing: what goes into
the features has mattered more than what is computed over them, every time -- and
which features help is task-specific enough that a single-dataset result should not be
generalised, in either direction.

### Does any of this help TabICL? (RelBench rel-f1)

Everything above is infrastructure; the synthetic relational benchmark only shows the
plumbing works, because the signal was planted there deliberately. `eval_relbench.py`
runs the real question on RelBench's rel-f1 / driver-top3 -- predict a top-3 finish
from a timestamped prediction table over a genuine multi-table schema, features built
under a per-row cutoff so nothing leaks:

| features | count | TabICL AUC | GBDT AUC |
|---|---:|---:|---:|
| entity table only | 6 | 0.5626 | 0.4123 |
| + relational history | 147 | **0.8125** | 0.7467 |

**+0.25 AUC.** The entity table alone is near chance, which is the point: all of the
lift comes from history the flattening aggregates. TabICL also beats a gradient-boosted
baseline on the identical feature matrix (0.8125 vs 0.7467), so the features are not
merely feeding a stronger model.

**What this run exposed.** RelBench's shape is one row per *(entity, prediction time)*
-- rel-f1 averages ~15 rows per driver and reaches 59 -- and the first implementation
assumed one row per entity key. It did not merely lose accuracy; it could not run at
all, because mapping a cutoff through a duplicated index raises. Aggregation is now per
entity *row*: child rows join to entity rows, are filtered against that row's own
cutoff, and group by row position. Cost is `|child| x (rows sharing a key)`, which is
the price of per-row-correct features.

Nested children plus repeated entity keys is rejected rather than approximated: a
grandchild's cutoff is genuinely ambiguous when the same key appears at several
prediction times.

### Do the graph features help? (RelBench rel-event)

rel-f1 has no entity-to-entity graph, so motifs were tested separately on rel-event /
user-ignore, which carries a real user-user friendship table. `eval_relbench_motif.py`,
three nested feature sets:

| features | count | TabICL AUC | GBDT AUC |
|---|---:|---:|---:|
| entity table only | 6 | 0.5665 | 0.5947 |
| + relational history | 32 | 0.5977 | 0.6272 |
| + motif features | 35 | **0.7332** | 0.7304 |

**+0.135 AUC from three columns** -- degree, triangles, clustering -- which is more
than the 26 relational aggregate columns added (+0.031). 91.8% of task users appear in
the graph; 68,085 triangles over 213,703 edges computed in 0.9 s.

Two honest qualifications. `user_friends` has no timestamp, so the graph is static:
RelBench models it that way, but a friendship formed after a prediction time is still
visible to that prediction, making this an upper bound on a strictly causal version.
And on this task TabICL and GBDT end up level (0.7332 vs 0.7304) -- the features carry
the win, not the model.

**Which column actually earns it.** Ablating (n_estimators=1, so absolute numbers
sit slightly below the table above; the comparison is internally consistent):

| features | AUC | delta |
|---|---:|---:|
| relational only | 0.6032 | — |
| + degree | 0.7183 | **+0.115** |
| + degree, triangles | 0.7345 | +0.016 |
| + all three | 0.7321 | -0.002 |
| + triangles only | 0.6742 | +0.071 |
| + clustering only | 0.6832 | +0.080 |

**`degree` -- a one-line `np.bincount` -- captures ~85% of the lift.** Triangles add
+0.016 on top of it, and clustering adds nothing once both are present, which is
expected since it is a deterministic function of them.

So on this task the compiled WCOJ, the semiring layer and the FAQ elimination are
collectively worth about +0.016 AUC. That is a real gain and it is not nothing, but it
is an order of magnitude less than the free feature, and the honest reading is that
the join machinery is not what pays off here. Triangles are worth more in isolation
(+0.071) than alongside degree, i.e. much of what they carry is degree in disguise.

Where the machinery would justify itself is a task whose signal is genuinely local
density -- fraud rings, collusion, mutual-connection patterns -- rather than
popularity. That has not been demonstrated, and should not be assumed from this.

**A bug this found.** `user_friends` is 30.4M rows of which all but 213,703 have null
keys. `pd.factorize` maps nulls to -1, and the counting kernels accumulate into arrays
indexed *by value*, so a negative value was an out-of-bounds write rather than a wrong
answer. Values are now validated before reaching the kernel. Any real table with a
nullable foreign key would have hit this.

### Two ways past the degree confound: typed and temporal motifs

"Degree carries 85%" is not an accident of this task. Motif counts are partly
*determined* by the degree sequence -- conservation laws relating subgraph counts
follow directly from conserving degrees (Bhat et al., "Motif conservation laws for
the configuration model", 2014) -- and orbits are formally redundant with each other,
degree being orbit 0 (Yaveroğlu et al.: 4 of the 15 orbits up to 4 nodes are
computable from the rest). A k-star count is the extreme case: it is exactly
`C(d, k)`, so it carries no information a model cannot already get from `d`.

Going to *larger* undirected motifs therefore buys little. Two refinements are not
recoverable from degree, and both are now implemented.

**`typed_triangle_counts` / `typed_motif_features`.** Triangles split by the multiset
of edge types they use, which is the vertex-collocation-profile argument
(Lichtenwalter & Chawla, WWW 2012): the discriminative power comes from distinguishing
isomorphism classes by relation type, not from more nodes.

The counting argument is what makes it cheap and exact. Under `a < b < c` a triangle
occupies three determined slots -- `(a,b)`, `(b,c)`, `(a,c)` -- so each slot's type is
determined too. Run the join once per *ordered* type triple, bucket by the sorted
triple, and every triangle is counted exactly once with nothing to correct
afterwards. That is `k**3` joins for `k` types, each over a subset of the edges;
`MAX_EDGE_TYPES = 6` guards the cube. Types are assumed disjoint, and the test suite
asserts the census sums back to the untyped count, which is what makes this a
refinement rather than a different feature.

**`temporal_motif_features`.** Two separate things that are worth not conflating:

* *Causality.* Features are computed from edges **strictly before** each row's cutoff,
  matching `flatten_relational`, so both feature families share one notion of time.
  This is what removes the caveat above rather than restating it.
* *Recency and ordering.* `windows` restricts to a lookback, and `n_phases` splits each
  window into equal-duration phases and adds the ordered census over them. This is the
  part degree provably cannot fake: a triangle whose edges appeared within a week is
  not the triangle that took three years, and `p0_p0_p1` -- two edges early, one late
  -- is triadic closure caught in the act rather than inferred. Paranjape, Benson &
  Leskovec (WSDM 2017) is the reference; the gains reported for temporal motifs are the
  ones that most consistently survive a degree baseline. That is the motivation, and it
  is worth reading against the measurement below, where the cutoff earns its place and
  the ordering census does not.

Phases reduce to types -- a phase is an edge type that happens to be a time bucket --
so the ordering census is `typed_triangle_counts` over phase-partitioned edges and
needs no separate kernel. Cost is one motif computation per (distinct cutoff x window),
times `n_phases**3` joins. Distinct *cutoffs*, not rows, which is what makes it
affordable: RelBench task tables share a handful of prediction timestamps across many
rows.

Measured on one hub-skewed graph -- 235,657 undirected edges, 30k nodes, 369,754
triangles -- reporting every node, compiled backend, same machine as the table above:

| | time | columns | check |
|---|---:|---:|---|
| untyped `motif_features` | 1.48 s | 3 | — |
| `typed_motif_features`, k=2 | 1.83 s | 6 | census == untyped total |
| `typed_motif_features`, k=3 | 2.54 s | 13 | census == untyped total |
| `temporal_motif_features`, 2 windows, 1 phase | 1.96 s | 6 | — |
| `temporal_motif_features`, 2 windows, 2 phases | 3.15 s | 14 | census == window total |
| `temporal_motif_features`, 2 windows, 3 phases | 4.24 s | 26 | census == window total |

The `k**3` and `n_phases**3` growth is real but mild at this scale, because each join
runs over a `1/k` slice of the edges; the fixed cost of symmetrising and deduping the
edge list is a large share of the untyped 1.48 s baseline.

#### Making the census fast: the cost was never the language

On a rel-event-shaped synthetic -- three relations of 213k/72k/16k edges, 23 cutoffs,
19k query rows -- the obvious implementation took 57.3 s. Profiling the first cached
version put 80% of the remaining time inside the compiled join, which reads as "already
optimal, and what is left is Python". Both halves of that reading were wrong.

| | time | what changed |
|---|---:|---|
| slice every type at every cutoff, full census each time | 57.3 s | — |
| hoist static types out of the cutoff loop | 25.7 s | Python |
| dedupe identical tries within a call, drop the identity-permutation copy | 19.2 s | C++ / Python |
| reuse a built index across queries (`Relation` handles) | 8.1 s | C++ |
| pack `(u << 32) \| v` instead of `np.unique(axis=0)` | **6.7 s** | Python |

**8.6x end to end, and the join kernel itself was never touched.** Every win was
redundant work, not slow code:

* *Static types are loop-invariant.* Their symmetrisation, degrees, and any triangle
  whose three slots are all static do not depend on the cutoff. Level I was re-deriving
  the 213k-edge friendship graph 23 times per split.
* *The index was rebuilt per query.* A `k=3` census issues 27 queries over 3 tables and
  was building 81 tries, each a full lexicographic sort. `Relation` handles make that 3.
  This alone took the in-kernel time from 20.4 s to 5.4 s -- so **73% of what profiled as
  "join time" was index construction, not search.**
* *Fancy indexing copies.* `atom.data[:, cols]` gave each of a triangle's three atoms a
  private copy of the same edge table, which also hid from the backend that they were the
  same buffer. Skipping the copy when the permutation is the identity fixed both.
* *`np.unique(axis=0)` sorts a void view of each row*, several times slower than sorting
  int64. Node ids fit in 32 bits, so packing the pair into one key is exact, with a
  fallback for ids that do not fit.

Two correctness notes, because both were nearly lost in the speedup. Trie dedup keys on
`(pointer, rows, arity)` and **not** on the pointer alone: `edges[0:stop]` for two
different stops shares a base address while denoting different relations, which the
temporal path produces routinely. And the handle path initially bypassed
`_require_non_negative`, which is the guard against the out-of-bounds write recorded
above; it now runs once per census instead of once per query. Both are tested.

All nine eval levels return bit-identical AUCs before and after.

#### Which of them earns it (RelBench rel-event again)

rel-event has three user-user relations, not the one the earlier run used:

| relation | edges | task users | timestamped |
|---|---:|---:|---|
| `friend` (`user_friends`) | 213,703 | 91.8% | **no** |
| `coinvite` (same event, invited) | 72,383 | 78.0% | yes |
| `coattend` (same event, yes/maybe) | 16,589 | 44.1% | yes |

Co-occurrence edges are capped at 40 users per event -- a 10,000-invitee event
contributes 50M pairs and nothing discriminative -- and dated by the event's
`start_time`. That is deliberately conservative: the invitation was sent *before* the
event, so the edge is credited later than it truly formed, which under-uses information
and cannot leak.

`eval_relbench_typed_temporal.py`, same task and split as the table above:

| | relations | causal | typed | features | TabICL AUC | GBDT AUC |
|---|---|---|---|---:|---:|---:|
| A relational only | — | — | — | 32 | 0.6032 | 0.6509 |
| B + static friend motifs | friend | no | no | 35 | 0.7332 | 0.7258 |
| G + static co-occurrence | co-occ | no | no | 35 | 0.8166 | 0.7985 |
| D + causal motifs | co-occ | yes | no | 35 | 0.8056 | 0.7604 |
| H + typed causal | co-occ | **yes** | yes | 38 | **0.8261** | 0.7663 |
| I + typed, friend static | all 3 | partly | yes | 45 | 0.8420 | 0.7832 |
| C + typed static motifs | all 3 | no | yes | 45 | **0.8591** | 0.8174 |
| E + causal, windowed, phased | co-occ | yes | no | 46 | 0.7921 | 0.7559 |
| F + static friend AND causal | all 3 | partly | no | 49 | 0.8162 | 0.7477 |

B reproduces the 0.7332 recorded above, so this is the same measurement as before.

**The decomposition, because no single comparison isolates anything.** B and C differ
in three ways at once -- relation set, typing, and cutoff -- so the gap between them
attributes nothing. Changing one thing at a time from B to C:

| step | change | AUC | delta |
|---|---|---:|---:|
| B | untyped, friend, static | 0.7332 | — |
| G | -> co-occurrence relations | 0.8166 | **+0.083** relation choice |
| D | -> enforce the cutoff | 0.8056 | **-0.011** causality |
| H | -> split by type | 0.8261 | **+0.021** typing |
| I | -> add friend as a static type | 0.8420 | **+0.016** relation count |
| C | -> drop the cutoff again | 0.8591 | **+0.017** the leak |

Sums to +0.126 exactly.

**Type splitting earns its place, but it is worth +0.021, not +0.126.** The clean
comparison is D -> H: same edges, same cutoff, typed versus not. The larger figure is
what you get by also swapping the relation and re-introducing the leak, and reading it
as a typing result would repeat precisely the error level G was added to catch one step
earlier. Typing is real, cheap and not available to degree -- and it is a fifth the
size of simply choosing a better relation.

**Relation choice dominates everything else here: +0.083.** Co-invitation and
co-attendance beat declared friendship by more than typing, causality and windowing
combined. Worth remembering before reaching for the join engine: the largest single win
in this table came from picking different edges, not from computing more on the same
ones.

**Causality costs about -0.011 to -0.017 and should be paid.** G -> D isolates it under
no typing, C -> I under typing. Both are small and in the expected direction, since
removing future information should lose a little. The useful conclusion is that the
leak recorded against the original result was not propping it up. H (0.8261) is the
best strictly causal configuration: every relation in it carries a timestamp, it beats
the original leaky 0.7332 by +0.093, and it is correct. C's 0.8591 is higher and leaks.
I sits between them and leaks too -- friendship has no timestamp, so including it as a
static type makes the whole result only as causal as its least causal relation.

**Windows and phases do not earn it: -0.014 (D -> E).** Eleven extra columns for a loss.
The recency/ordering half of the temporal machinery is a negative result on this task.
`p0_p0_p1` is a well-motivated feature and the census is exact, but rel-event's
prediction window is five months against event timestamps that only reach 13.9% coverage
by the midpoint cutoff, so most rows see too little history for a phase split to mean
anything. A denser, longer-running temporal graph is where it would get a fair test.
`typed_temporal_motif_features` therefore offers no phase splitting: combining phases
with types would cost `(k * n_phases) ** 3` joins for a measured loss.

**What the join engine is worth, restated.** The earlier ablation put it at +0.016
(degree vs degree+triangles). Typing raises that to +0.021 on the same causal footing,
still an order of magnitude below what free features and relation choice deliver. The
machinery is correct, fast and general; on this task it remains the small term.

#### Screened on a second graph: rel-arxiv confirms the diagnosis and not the hope

The result above is one task, and "a denser, longer-running temporal graph is where it would
get a fair test" was left standing as a claim. `screen_motif_signal.py` tests it cheaply --
a triangle census and a GBDT, no TabICL fit -- with the bar **pre-registered at rel-event's
+0.016**, the gain already measured and already declined. rel-arxiv/paper-citation, 1,223,361
citation edges dated by `Submission_Date`, 9 distinct cutoffs, 100k rows per split, base rate
0.475:

| features | val AUC | gain |
|---|---:|---:|
| degree alone | 0.8049 | — |
| + triangles, clustering | 0.8045 | **−0.0004** (sd 0.0003, 0/3) |
| + ordered phase census | 0.8072 | **+0.0027** (3/3) |

**Triangles add nothing over degree: −0.0004, on a graph where they plainly exist.** 82.2% of
rows have an edge before their cutoff and **51.6% have a triangle**, so this is not a null for
want of structure -- the census found triangles for half the rows and a GBDT could not use
them once it had the degree. That is the same finding as rel-event (+0.016 there, which this
does not even reach) on a graph an order of magnitude denser in the relevant sense. **VERDICT:
FAIL. No rel-arxiv round.**

**Read as weak evidence, because of what the label is.** "Will this paper be cited in the next
six months" is preferential attachment: the target *is* popularity, and degree alone reaching
0.805 says so. A task whose label is degree cannot ask whether density matters where
popularity does not. The pre-registration said this before the run, and it is the reason the
failure does not close the question -- `rel-arxiv/author-category` is the better-matched label
(co-author community predicting research area) and needs multiclass support this harness lacks.

**The phase census flipped sign, which is the diagnosis confirming.** −0.014 on rel-event,
**+0.0027** here, positive on all three fits. The stated reason for rel-event's loss was that
its timestamps reach 13.9% coverage by the midpoint cutoff, leaving most rows too little
history for a phase split to mean anything; on a graph with real edge times the sign reverses.
**It is +1.9 sampling SE on a 100k validation split, so it is suggestive and not resolved** --
and the three replicates vary the GBDT seed, not the sample, so their sd of 0.0003 measures
model variance and understates the real uncertainty. What it does establish is that the
rel-event result was a coverage artefact rather than the ordering census being worthless. It
is still a fifth of the bar.

### Cyclic patterns: worst-case optimal joins

Tree aggregation covers parent-child schemas, which is what RelBench uses. It cannot
express features that live on a *cycle* -- triangles, mutual counterparties,
clustering coefficients -- because there is no root to roll up from, and a cyclic
join computed with binary joins materialises an intermediate that can be
quadratically larger than the answer.

`_wcoj.py` implements leapfrog triejoin (Veldhuizen, ICDT 2014; Ngo-Porat-Re-Rudra,
PODS 2012), whose cost is bounded by the AGM bound of the query rather than by any
intermediate. `motif_features` exposes degree / triangle count / clustering
coefficient per node, ready to concatenate onto a flattened feature table.

**Two implementations.** A pure-Python leapfrog triejoin, and an optional compiled
hash join built by `python -m tabicl.scaling.build_native` (needs a C++17 compiler and
`pybind11`; on Windows that means MSVC Build Tools -- clang alone cannot link a
CPython extension because it ships no Windows SDK). `wcoj_join` picks the compiled one
when present.

**The Python one is not competitive, and comparing it to `pandas` measures the
language, not the join.** Timing triangle counting, all single-threaded (verified by
cpu-time/wall-time ~= 1 for each):

| graph | triangles | LFTJ (py) | pandas | scipy A^3 | **compiled WCOJ** |
|---|---:|---:|---:|---:|---:|
| n=400, p=0.20 | 84837 | 1.803 s | 0.051 s | 0.030 s | **0.011 s** |
| n=800, p=0.05 | 10458 | 0.747 s | 0.033 s | 0.061 s | **0.007 s** |
| n=1500, p=0.02 | 4464 | — | 0.039 s | 0.110 s | **0.010 s** |
| n=2000, p=0.02 | 10665 | — | 0.098 s | 0.267 s | **0.021 s** |

Compiled, the join wins everywhere: 2.8-4.8x over `pandas`, 2.1-12.7x over `scipy`,
and 110-167x over the same algorithm in Python. On hub-skewed graphs `scipy` collapses
(3.2 s vs 0.12 s) because `A^3` densifies, while `pandas` stays competitive -- its
`u < v` orientation happens to prune the star, so those graphs are not the AGM
showcase they look like.

So `triangle_counts` prefers the compiled join and falls back to sparse `diag(A^3)/2`.

**Parallelism.** Only the compiled join is threaded; `pandas` `merge` and `scipy`
sparse matmul are both serial by design. Cost model first, scheduler second:

* Candidate cost is the **product** of touching relations' bucket sizes, not the min.
  Relations sharing only a bound variable fan out multiplicatively, and a min estimate
  underweights exactly the hub vertices that dominate.
* Scheduling is a shared atomic cursor over a **cost-descending** list. Round-robin
  over value-sorted candidates is systematically unfair (degree correlates with value
  globally); static partitioning strands a thread whose batch outruns its estimate.
* Items above a fair share (`total / threads`) are **pre-split** recursively, because
  one oversized item runs start-to-finish on one thread and caps speedup regardless of
  how well the rest is packed.

Measured on 8 logical cores:

| query | 1 thread | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| triangles, n=1400 p=0.12 (791k out) | 1.00x | 1.35x | 1.84x | 2.08x |
| 4-clique, n=400 p=0.20 (67k out) | 1.00x | 1.59x | 2.52x | 2.94x |

Sub-linear, honestly: at 20-50 ms per query the serial trie build, the per-thread
output merge, and thread startup are a real fraction of the total, and the
output-heavy triangle query is worse than the search-heavy clique one. These are small
workloads; the serial fraction shrinks as inputs grow.

### FAQ: counting without enumerating

`wcoj_count` takes the FAQ view (Abo Khamis, Ngo, Rudra, PODS 2016) -- aggregation is
variable elimination over a semiring -- and accumulates during the join instead of
enumerating tuples and counting afterwards. Once a prefix is fixed, the number of
completions is the size of an intersection, known without visiting its elements, so
every already-bound variable is credited in O(1).

| graph | triangles | enumerate | count |
|---|---:|---:|---:|
| n=1000, p=0.10 | 167782 | 0.0223 s | **0.0179 s** |
| n=1400, p=0.12 | 791891 | 0.0637 s | **0.0488 s** |
| n=1800, p=0.10 | 972916 | 0.0981 s | **0.0660 s** |

**1.2-1.5x, and a prediction of mine that did not survive contact.** I expected this to
lift the parallel ceiling too, on the theory that per-thread output buffers and the
final merge were the serial bottleneck. They are not: counting scales no better than
enumerating (1.5-1.8x on 8 threads either way). The dominant cost is the intersection
work itself, which both paths do identically. What FAQ actually buys here is a modest
constant factor plus O(1) rather than O(output) memory -- 23 MB of result tuples never
allocated on the largest case above.

A total-only variant that sizes the intersection by merging without storing it was
built and then removed: it measured *slower* than the per-value path (0.077 s vs
0.066 s), because avoiding one materialisation cost two others. Not worth a second
code path.

`triangle_counts` uses the counting path.

### Semirings, and FAQ over them

`_semiring.py` names the algebra both aggregation paths were special cases of. A
semiring's `add` combines *alternatives*, `mul` combines *independent parts*;
aggregation over a join is variable elimination in one of them, which is why counting
a pattern, summing weights over it, and finding its cheapest witness are one algorithm
with three operators rather than three algorithms.

| semiring | add | mul | answers | invertible |
|---|---|---|---|---|
| `SUM_PRODUCT` | `+` | `x` | how much / how many | **yes** |
| `MIN_PLUS` | `min` | `+` | cheapest witness | no |
| `MAX_PLUS` | `max` | `+` | strongest witness | no |
| `BOOLEAN` | `or` | `and` | does one exist | no |

`invertible` is not decoration: it is exactly the line between statistics that survive
incremental maintenance and ones that need a rebuild. Counts and sums can be
un-added when a row is deleted; removing the current minimum tells you nothing about
the next one. That is why `_relational.py` carries `count/sum/sumsq` and derives
`mean`/`std`, rather than storing them.

`check_semiring_laws` verifies the axioms on samples. Worth running on a custom
semiring: one that fails distributivity still produces numbers during elimination,
they are just silently the wrong numbers.

**What this unlocked.** The tree roll-up only ever applied `add` -- it sums child
sums, mins child mins. Combining *across a hop* had no expression at all: a per-order
discount times that order's item total, a probability along a chain. `hop_product`
supplies `mul`.

`wcoj_aggregate` is the cyclic counterpart, and full FAQ: `mul` accumulates along a
witness, `add` combines alternative witnesses, folded during elimination so no witness
is materialised. Counting is the case where every payload is 1.

```python
# cheapest triangle each node sits in, over node weights
total, overall, per_node = wcoj_aggregate(
    atoms, ["a", "b", "c"], weights, semiring=MIN_PLUS, less_than=[("a", "b"), ("b", "c")]
)
```

This is what tree aggregation genuinely cannot do: a cyclic pattern has no root to
roll up from, so "the cheapest triangle containing v" has no parent-child formulation
at all. Verified against brute-force enumeration for all three rings, on the overall
aggregate and per node.

The compiled kernel implements `SUM_PRODUCT`, `MIN_PLUS` and `MAX_PLUS`; a custom
semiring raises rather than silently falling back to the wrong operator.

Verified against brute-force enumeration on random graphs up to density 0.9, on
4-cycles, and per-node counts against exhaustive triple enumeration.

**Not decomposable, and not faked:** `nunique` and `mode`. Exact distinct-count over
a join needs a sketch (HyperLogLog and friends); a mode of modes is not a mode. At
depth 1 both are exact. Deeper, `nunique` becomes "distinct values per parent,
summarised again", which is a different quantity -- documented rather than silently
wrong.

## 4. Test-time compute

TabPFN-3's "Thinking mode" mechanism is **undisclosed** — §2.6 says only that it
"applies additional inference-time computation", and it ships API-only. There is
nothing to copy, so this is an original design in the same spirit.

**Scope.** Freeze the backbone, extract row representations, and spend extra
inference compute on top:

- refit a light head (logistic / ridge) on frozen train representations;
- blend with the model's own ICL output, weight chosen on a held-out split;
- optionally average over several estimator permutations.

Compute knob = number of refits/permutations. Guarded by a validation split so it
cannot regress below the base model.

## Benchmark (12 GB RTX 3060)

`bench.py` measures `torch.cuda.max_memory_allocated`, wall clock, and accuracy.

- Row-chunk: sweep `n_train` x `n_features`, report peak VRAM and the OOM threshold
  crossed by chunking; assert output equivalence.
- MQA: cache bytes and accuracy delta.
- Relational: AUC vs. a single-table baseline on a synthetic 3-table schema.
- TTC: accuracy gain vs. added seconds.

Sized to run in minutes, not hours.

## Measured results (RTX 3060, 12 GB)

### 1. Row-chunked column embedding — works, ~6x

`n_features=100`, `chunk_size=8192`, `col_chunk_size=32`, fp32:

| rows | unchunked MiB | chunked MiB | saving | unchunked s | chunked s |
|-----:|--------------:|------------:|-------:|------------:|----------:|
| 4000 | 1781 | 574 | 3.10x | 0.56 | 0.42 |
| 8000 | 3530 | 1138 | 3.10x | 0.47 | 0.64 |
| 16000 | 7047 | 1165 | 6.05x | 0.88 | 1.23 |
| 32000 | 14078 | 2265 | 6.22x | 15.90 | 2.39 |

Consistent with TabPFN-3's reported ~5x. Output matches unchunked to ~1e-5 (fp32
noise); tests assert 1e-4.

At 32k rows the unchunked path needs 14 GB on a 12 GB card. On Windows it does not
raise OOM — WDDM silently spills to host memory — so the cost shows up as a 6.6x
*slowdown* rather than a crash. Chunking keeps it resident. On Linux the same shape
would OOM outright.

Chunking both axes is what matters. Row-only chunking capped at ~2x, because phase A
still ran the full stack over every context row and its feed-forward intermediate
became the new peak — exactly why the paper chunks phase (i) over columns too.

### 1b. L40S (46 GB, Linux) — clean OOM, and vs. offload

The 3060 numbers could not show a true OOM (Windows spills to host RAM instead) and
never compared against `offload`. Both are settled here.

**Isolated column-embedding stage** (`n_features=100`, `chunk_size=8192`):

| rows | unchunked MiB | chunked MiB | saving | unchunked s | chunked s |
|-----:|--------------:|------------:|-------:|------------:|----------:|
| 16000 | 7047 | 1165 | 6.05x | 0.21 | 0.28 |
| 64000 | 28140 | 4515 | 6.23x | 0.96 | 1.09 |
| 128000 | **OOM** | 9015 | — | — | 2.18 |

Chunking turns an OOM into a 9 GB run. **Correcting the 3060 write-up: chunking is
not faster.** It costs ~13% wall clock at 64k. The earlier "6.6x faster" was purely
the Windows paging artifact; the honest overhead matches the paper's "a few percent".

**End-to-end `predict_proba`, vs. offload.** This is the comparison that matters,
since offload is what TabICLv2 actually does at scale:

*n_train=60000, n_test=6000:*

| variant | peak MiB | seconds |
|---|---:|---:|
| naive | 21763 | 2.47 |
| offload=cpu | 21755 | 3.53 |
| row_chunk | **15775** | **2.20** |
| offload + row_chunk | 15367 | 3.64 |

At this size **offload is useless** — it reclaims 8 MiB (0.04%) for 43% more time,
because it moves *outputs* while the peak is *activations*. Chunking cuts 27% and is
slightly faster than naive.

*n_train=150000, n_test=10000:*

| variant | peak MiB | seconds |
|---|---:|---:|
| naive | 31138 | 6.92 |
| offload=cpu | 23004 | 17.27 |
| row_chunk | 26445 | 6.90 |
| offload + row_chunk | **19422** | **9.25** |

Offload finally earns its keep on memory (-26%) but costs 2.5x wall clock — the
slowdown TabPFN-3 criticises. **offload + row_chunk Pareto-dominates offload alone:
16% less memory *and* 46% less time.** That is the result that justifies the port.

*n_train=300000, n_test=10000:*

| variant | peak MiB | seconds |
|---|---:|---:|
| naive | 38820 | 18.57 |
| offload=cpu | 23069 | 38.42 |
| row_chunk | 38198 | 19.39 |
| offload + row_chunk | **22456** | 23.36 |

Beyond ~150k rows the ICL stage, not the column embedder, owns the peak, so chunking
alone contributes little (1.6%); the combination is what wins.

**A bug this found.** At 300k, chunking alone first measured *worse* than naive
(39939 vs 38820 MiB) — `_run_tf_col` was not passing `inplace`, so it allocated a
full extra `(N, C, d)` output buffer that cancelled the saving and then some. Fixed
by threading `inplace=True` through the two call sites that do not reuse the buffer
(the mixed-radix loop reuses `src_with_y` across digits and must stay out of place).
Worth 1.7 GB at 300k. Only visible at a scale the 3060 cannot reach.

### Chunking the ICL stage -- a null result

Beyond ~150k rows the ICL stack owns the peak, so it was chunked too. Within a block
every row is a query but only the first `train_size` rows supply K/V, so the same
two-phase trick applies: harvest each block's K/V once, then stream queries.

Isolated, it works, and the numbers look excellent (RTX 3060, `d_model=512`, 12 blocks):

| rows | unchunked | chunked | saving |
|---:|---:|---:|---:|
| 20000 | 364 MiB | 104 MiB | 3.50x |
| 100000 | 1761 MiB | 490 MiB | 3.60x |
| 200000 | 3518 MiB | 977 MiB | 3.60x |

**End-to-end it buys exactly nothing** (L40S, 100 features, `predict_proba`):

| n_train | default | ICL chunked only | col chunked |
|---:|---:|---:|---:|
| 60000 | 22376 MiB | 22368 MiB | **15827 MiB** |
| 150000 | 31257 MiB | 31257 MiB | **24900 MiB** |
| 300000 | 38569 MiB | 38569 MiB | 38082 MiB |

Identical, at every size, including the regime it was built for.

**Why, and it is a repeat offence.** `InferenceManager` already batches the ICL stage,
so the full-row forward the microbenchmark measured never happens in the real path.
The isolated figure came from calling `tf_icl` directly on one big tensor, bypassing
the manager. This is the *same* mistake as the column embedder's 6.2x isolated versus
1.38x end-to-end -- documented in this very file, and then repeated. A microbenchmark
that bypasses the caller measures a code path the model does not take.

`chunked_icl_encoder` is kept: it is correct, exact to ~4e-5, tested, guarded against
positional encoding, and would matter if the manager's batching were ever bypassed.
It stays **off by default**, and the isolated 3.6x should not be quoted as a win.

**Guidance:** `row_chunk` on `COL_CONFIG` is the one that pays -- up to ~1.4x, and
only below ~300k rows, above which the peak has moved elsewhere again. Combine with
`offload="cpu"` at scale. Leave `ICL_CONFIG`'s `row_chunk` off.

### The limit of row chunking: wide tables OOM before it is reached

Row chunking cannot rescue a *feature-count* explosion, and RelBench rel-event is the
case that showed it. 1,670 features over 21k rows wanted 35.7 GB and died with
`row_chunk` enabled — because the allocation happens **before** `tf_col` is ever
called, at `embedding.py:444`:

```python
src[..., :train_size, :] = src[..., :train_size, :] + y_emb
```

`src` is the full `(B, C, N, d)` activation, materialised by `_compute_embeddings`
and handed to the set transformer only afterwards. `row_chunked` swaps out `tf_col`,
so everything upstream of that swap runs unchanged. Wrapping a callee cannot bound a
tensor its caller already allocated.

This is a real boundary, not a tuning problem, and it splits the two failure modes:

| symptom | axis | fix |
|---|---|---|
| rel-event: 1,670 features, 21k rows | columns | `Table(max_columns=...)` — cut features *before* the model |
| rel-avito: 149 features, 152k rows | rows | `row_chunk` + `offload` — what they were built for |

`max_columns` caps source columns per table by non-null coverage. On rel-event it took
1,670 features to 200, feature time to ~5s, and turned "cannot run" into a real score.
It is deliberately blunt: coverage is a target-free proxy, so it cannot leak, but it
does not know which columns matter. Chunking the pre-`tf_col` path properly would mean
pushing the split into `_compute_embeddings` itself — tractable, since columns are
independent there too, but a change to the model rather than a wrapper around it.

Sweeping the cap on rel-event / user-ignore (official test split):

| `max_columns` | features | test ROC-AUC x100 |
|---:|---:|---:|
| 1 | 74 | **80.81** |
| 2 | 110 | 79.85 |
| 4 | 200 | 77.85 |
| unbounded | 1,670 | *OOM at 35.7 GB* |

Monotone on this task, and the tightest cap scored highest: 1,670 features down to 74
bought three points rather than costing them.

The sweep stops at 1, 2 and 4 deliberately. `max_columns=8` drove free RAM to 235 MB of
64 GB and began paging -- the same thrash that repeatedly killed this task before -- so
it was stopped rather than left to produce a number that would really be measuring the
page file.

#### The generalisation, which does not hold

An earlier version of this section read "the tightest cap wins" and "as many features as
fit is the wrong default". **That was one task generalised into a recommendation, and it
is wrong.** Running the same cap across the other tasks:

| task | uncapped | `max_columns=2` | effect |
|---|---:|---:|---:|
| rel-event / user-ignore | *OOM* | 79.85 | **enables the task at all** |
| rel-trial / study-outcome | 65.40 | 65.40 | 0.0 |
| rel-f1 / driver-top3 | 79.77 | 60.31 | **-19.5** |

A 22-point spread in the effect of one setting. rel-f1's three children are small and
entirely numeric -- 359 features over 1,941 rows, none of them redundant -- so capping
at two columns per table deletes most of the signal. rel-event's 1,670 features over a
2.5M-row table are mostly noise. The cap does not know the difference, because coverage
cannot.

The corrected reading: **`max_columns` is a rescue, not a default.** Reach for it when a
schema cannot otherwise run, and do not tighten it on a schema that already fits. The
right value is task-dependent by a margin far larger than any effect measured elsewhere
in this file, so it should be chosen on a validation split rather than assumed -- the
same anchor-style calibration that the sampling literature recommends for context size.

This is the third time in this project that a single-task result did not survive a
second dataset (windows: +0.089 on rel-f1, median +0.001 elsewhere; the "2.5-point
categorical gap" that turned out to be a three-variable comparison). The pattern is
consistent enough to be a rule: **no lever goes in as a recommendation on one task.**

### Categorical statistics on the as-of scan -- built, correct, and not a win

The as-of path was numeric-only. Diffing what the two paths emit located the gap
exactly: identical source columns and identical numeric families, with the join path
additionally producing `mode` on every block and `nunique` on windows. At matched
relation breadth on rel-trial that was worth 2.21 (65.40 as-of vs 67.61 join).

Two blocks were added, on the argument that neither needed a sketch:

* **Top-K category histograms.** Fix a global codebook of the most frequent values,
  emit one indicator per category, prefix-sum it. An indicator is a counter, and
  counters are what this scan already carries -- so the block is exact rather than
  approximate and extends to windows for free, which `mode` cannot do.
* **Prefix mode.** The claim that `mode` "has no linear range algorithm at all" holds
  for a *range* and not for a *prefix*. Counts rise by one, so the leader changes only
  when a count strictly exceeds every count before it, and the row that does it holds
  the new mode. Three segmented scans, no Python loop.

Both are exact and tested. Neither should be turned on by default:

| task | baseline | + histogram + mode | effect |
|---|---:|---:|---:|
| rel-trial / study-outcome | 65.40 | 66.65 | **+1.25** |
| rel-f1 / driver-top3 | 60.31 | 60.31 | 0.00 (children are all numeric) |
| rel-event / user-ignore | 79.85 | **76.64** | **-3.21** |

> **Resolved.** The -3.21 was correct, but only at `n_estimators=4`. The gap is ensemble-
> dependent -- +2.87 at 1, -0.95 at 2, -3.52 at 4, -3.06 at 8 -- so the apparently
> contradictory measurements were each right in their own regime. See "A configuration's
> sign flips with ensemble size" below. The conclusion stands at deployment ensemble
> sizes, which is what off-by-default is chosen for.

**It hurts rel-event by more than it helps rel-trial**, and costs 2.3x the model time
there (403s -> 912s). The explanation is consistent with everything else measured here:
the block's price is paid in *columns*, and rel-event is the task where column count is
already the binding constraint -- its cap sweep improved monotonically as features
shrank, 1,670 to 74. Adding columns loses there whatever information they carry.

Isolating the two on rel-trial also inverts the order they were proposed in:

| block | rel-trial |
|---|---:|
| mode only | -0.33 |
| histogram only | +0.94 |
| both | +1.25 |

`mode` alone is *negative*; it only pays as a complement, and +0.31 marginal is inside
the noise. The block built second on the cleaner theoretical argument is the weaker one.

Kept because it is correct, exact, tested, and the right tool when a schema's signal
really is categorical -- `top_k_categories` and `include_mode` are both **off by
default**. Not kept as a recommendation. The remaining ~0.96 to the join path on
rel-trial is windowed `nunique`, which needs an offline dominance count; on this
evidence that is not worth building.

One incidental fix: the join path resolved a tied `mode` with a non-stable sort, so a
tie could resolve differently between runs on identical data. Now stable.

### Context size: the curve is task-dependent too

A TabICL context is quadratic in compute and linear in peak memory, so shrinking it is
the cheapest lever on both. An independent evaluation across four temporal relational
databases reported curves shallow enough that 10k-20k rows land within epsilon of the
best score, and concluded 10k-20k is enough "on every relational dataset".

That does not hold here. Two tasks, official test split:

| context rows | rel-avito (full = 116,598) | rel-trial (full = 12,954) |
|---:|---:|---:|
| 1,000 | -- | 61.39 (-4.01) |
| 3,000 | -- | 62.66 (-2.74) |
| 5,000 | 63.30 (-1.16) | -- |
| 10,000 | **64.85 (+0.39)** | -- |
| full | 64.46 | 65.40 |

**rel-avito's curve is flat and rel-trial's is not.** 10,000 rows -- 8.6% of rel-avito's
context -- *matched* the full set, while rel-trial gives up 2.74 at 23% of its context.
Same lever, opposite verdicts, which is the third setting in this file to behave that way.

The +0.39 by which 10,000 rows exceeded the full context is inside the measured noise
band (paired-gap sd 0.29, so +-0.58), so it is a tie rather than a win. That does not
weaken the practical point -- matching full context on 8.6% of the rows is the same
result for cost purposes -- but "beat" was an overstatement of a difference the
measurement cannot resolve.

Two things follow. First, rel-avito never needed a GPU: 10k context on CPU scores better
than 116,598 rows on an L40S, so `row_chunk` and `offload` are one cost lever and "use
fewer rows" is a cheaper one that was never tried. Second, "10k-20k is enough" is not a
default worth adopting -- it is a hypothesis to test per dataset, which is what
`calibrate_context_size` is for.

Plausibly the difference is pool size: rel-avito has 116k label rows and heavy
redundancy among them, rel-trial has 12,954 and little. That predicts the curve flattens
where the context is large relative to the task's diversity, and is worth checking
before it is believed.

Measurement note: the 20,000 arm of rel-avito was stopped rather than run. Model time at
5,000 was 1,369s on CPU and the cost is quadratic, so it was ~6 hours for one point on a
curve whose shape was already clear. Running it concurrently with the rel-trial sweep
also drove committed memory to 99 GB against 64 GB physical and began paging -- the same
thrash recorded elsewhere in this file, caused the same way, by running two heavy jobs
at once. rel-trial's 6,000 arm was lost to it.

### How much difference is real? A noise floor

Every comparison in this file is *paired*: same seed, same data, one setting changed.
That matters more than it sounds. Measured on rel-event across five seeds:

| quantity | mean | sd | range |
|---|---:|---:|---:|
| absolute score | 82.50 | 2.28 | 5.45 |
| paired gap (one setting changed) | +0.39 | 0.29 | 0.71 |

Pairing tightens the estimate **7.9x**. An unpaired comparison on this task could not
resolve anything below about five points; a paired one resolves about half a point.

Taking two paired standard deviations as a floor, **differences under about +-0.6 are not
measurable here**. Applied to the claims in this file: the `max_columns` effects
(-19.5, +3.0), the categorical blocks (+1.25, -3.21) and rel-avito's 5,000-row context
(-1.16) clear it; relation breadth (+0.35) and rel-avito's 10,000-row context (+0.39) do
not, and are recorded above as ties.

The floor is itself one measurement on one task, so it is an order-of-magnitude guide
rather than a threshold to apply mechanically.

### Calibrating per dataset -- better protocol, and not a cure

Three settings measured here reverse across datasets (`max_columns`, the categorical
blocks, context size), so the response was to stop looking for defaults and choose per
dataset on a validation split. `eval_relbench_calibrated` does that: feature spec and
context size selected on validation, test touched once.

It also fixes a protocol problem that had gone unremarked. The hand-picked table below
is a per-task *maximum selected on test* -- `max_columns` chosen by comparing 1/2/4 on
the test split, join-versus-scan the same way, context size the same way. That inflates
it, and it is not comparable with the published single-configuration figures beside it.

| task | hand-picked (selected on test) | calibrated (validation) | delta |
|---|---:|---:|---:|
| rel-f1 / driver-top3 | 79.77 | **80.70** | +0.93 |
| rel-event / user-ignore | 80.81 | **78.11** | -2.70 |
| rel-trial / study-outcome | 67.61 | **66.50** | -1.11 |

Mean 76.06 -> 75.10, so roughly **one point of the old table was test-selection
inflation** -- about what one would expect, and the calibrated column is the defensible
one.

**But calibration picked a configuration we already knew was wrong.** On rel-event,
validation preferred the categorical blocks (80.86 against 80.38 without) -- and those
blocks were independently measured at **-3.21 on test** for that exact task. The
validation split is 2,013 rows; rel-trial's is 960. At that size the selection signal is
comparable to its own noise, and a margin of 0.49 means nothing.

So "calibrate per dataset" is not the clean answer to "no default generalises". It is
better than selecting on test, and it is honest, but a noisy validation split can select
something worse than a sensible fixed default would have been. Two candidate fixes, both
untested at time of writing:

* **Repeated or reseeded splits** for selection rather than the single provided
  validation set. If rel-event's choice flips under reseeding, noise is confirmed as the
  cause.
* **A margin requirement before accepting a more expensive configuration.** rel-event's
  categorical block won by 0.49 on validation and cost 2.3x the compute. Requiring a
  real margin -- rather than the current tolerance, which only ever favours the cheaper
  option when scores are close -- would have rejected it.

rel-avito is the test of the noise explanation rather than another number: its validation
split is far larger, so if the diagnosis is right its calibration should behave better.

### AMP is on by default and costs 7.3 AUC on rel-event

The largest single effect measured in this project, and it is not a feature at all.

Chasing why two runners disagreed about `n_estimators` led to a much bigger discrepancy:
the same configuration scored 80.93 on CPU and 73.61 on CUDA. Isolated on one machine, one
variable at a time:

| ruled out | evidence |
|---|---|
| pandas | 3.0.5 and 2.3.3 give *identical* scores |
| torch version | pod torch 2.4.1 CPU = 80.93, local torch 2.12 CPU = 80.77 |
| checkpoint | byte-identical, `tabicl-classifier-v2-20260212.ckpt`, 110,368,038 B |

Leaving the device, and then the cause:

| configuration | rel-event / user-ignore |
|---|---:|
| CPU | 80.93 |
| CUDA, `use_amp=True` (**the default**) | **73.61** |
| CUDA, `use_amp=False` | **80.93** |

`use_amp=True` is set in all three inference configs. On synthetic well-separated data it
is harmless -- AUC 98.94 vs 98.93 -- but predictions still differ by `max|dp| = 1.8e-02`,
and *that* is the mechanism: on a task scoring ~80 the predictions cluster near the
boundary, so a perturbation of that size reshuffles enough pairwise orderings to cost
seven points. AMP is safe where the model is confident and expensive where it is not,
which is exactly backwards from where one would want to spend precision.

**This contaminates every GPU measurement in this file**, including rel-avito's 64.46 on
the L40S, which was run with the default and is therefore likely understated. It does not
touch the CPU results, and it does not touch the row-chunking equivalence proof -- that
was verified CPU-side at `max|dp| = 1.1e-05`.

Practical rule: **`use_amp=False` for any accuracy measurement on GPU.** AMP roughly
halves runtime (11s vs 20s here), so it remains reasonable for memory benchmarking and for
production ranking where a fixed threshold is not in play -- but not for a number anyone
will compare.

### A configuration's sign flips with ensemble size

The two runners that disagreed about the categorical blocks were both right. Measured on
a validated GPU host (AMP off; sanity cell reproduced the CPU baseline at 80.92 against
80.93), fit set held fixed at train+val, everything varying but `n_estimators`:

| `n_estimators` | plain | categorical | gap |
|---:|---:|---:|---:|
| 1 | 75.02 | 77.89 | **+2.87** |
| 2 | 79.65 | 78.70 | -0.95 |
| 4 | 80.92 | 77.40 | **-3.52** |
| 8 | 81.03 | 77.97 | -3.06 |

The categorical blocks help a single estimator and hurt an ensemble, crossing over between
1 and 2. So the original -3.21 was right at `n_estimators=4`, the +1.50 was right at 1, and
the disagreement was never noise or leakage.

The mechanism is visible in the columns. **Plain gains 6 points from ensembling (75.02 ->
81.03); categorical gains nothing (77.89 -> 77.97).** TabICL's ensemble members differ by
feature permutation, so a narrower feature set gives members that disagree usefully, while
the wider one appears to give members that are individually better but redundant. Ensemble
diversity, not raw feature quality, is what the extra columns cost.

**Consequence for this package.** `eval_relbench_calibrated` selected at
`--select-estimators 1` and scored the final fit at 4, purely to make the sweep affordable.
That is unsound whenever a setting's sign is ensemble-dependent, and it is exactly what
produced rel-event's calibrated 78.11: validation at `n_estimators=1` preferred the
categorical blocks, which really are better there, and the final fit at 4 then paid for a
choice made in a regime that does not predict deployment. `select_estimators` now defaults
to `n_estimators`.

Generalising: **calibrate in the regime you deploy in.** Any cheap-proxy shortcut during
selection -- smaller ensemble, smaller context, fewer seeds -- is only valid if it
preserves the *ranking* of candidates, and that is an assumption to test rather than
assume.

### Where this lands against published RelBench numbers

Official protocol throughout: fit on `train`, score the held-out `test` split, whose
labels require `mask_input_cols=False` to reveal. Test ROC-AUC x100. Comparison columns
are the published figures from the TabPFN-3 paper's Table 14.

| task | this branch | TabPFN-REL | RelGNN | RDBLearn+v3 |
|---|---:|---:|---:|---:|
| rel-f1 / driver-top3 | **79.77** | 79.98 | 85.69 | 82.72 |
| rel-event / user-ignore | 80.81 | 85.38 | 86.18 | 73.70 |
| rel-avito / user-visits | 64.46 | 66.68 | 66.18 | 66.76 |
| rel-trial / study-outcome | 67.61 | 76.43 | 71.24 | 72.89 |

Level with TabPFN-REL on rel-f1, -2.2 on rel-avito, -4.6 on rel-event, -8.8 on
rel-trial. Worth stating plainly: these are a generic flattening pipeline in front of a
stock TabICL, against systems built for relational data, and rel-event's figure only
exists because of the column budget above.

rel-avito is the case row chunking was actually built for, and it behaved that way. It
had OOM'd at 59 GB in attention on CPU -- 116,598 context rows x 176 features, a
row-axis problem, not a feature-count one. On an L40S with `row_chunk` and
`offload="auto"` the same configuration fits and the model fits in 90s. The as-of scan
built features over three children totalling 7.5M rows in 19s.

### Does schema depth help? A negative result on rel-trial

rel-trial / study-outcome is the one RelBench task where depth-2 is available today:
`flatten_relational` folds grandchildren only when entity keys are unique, and
rel-trial's are (11,994 rows, 11,994 distinct — rel-f1 and rel-event both repeat). It
also has a real two-level chain, `outcomes` (pk=`id`) <- `outcome_analyses`.

It is a favourable case on paper, because `outcomes` carries **no numeric columns at
all** — only `id` and `nct_id`, both keys. Every numeric quantity in that subtree
(`param_value`, `p_value`, `ci_*`) sits one level further down, so depth-2 is the only
way the schema reaches it.

| arm | features | test ROC-AUC x100 |
|---|---:|---:|
| depth-1 (`outcomes` only) | 21 | 49.12 |
| depth-2 (+ `outcome_analyses` folded per outcome) | 63 | 49.37 |

The folding is mechanically correct — 42 extra grandchild statistics materialise, and
this is the path where the mean-of-means bug lived, so it is also a check on the
factorized sufficient statistics against real data rather than a fixture. But both arms
sit at chance, and the extra depth moves nothing. Only 22.5% of outcomes have any
analysis, so most of those 42 columns are empty for most trials.

Read it narrowly: this says the `outcomes` subtree is uninformative for study-outcome,
not that depth never pays. It is one task, and it is the only one where the machinery
could be exercised at all. The result that *does* generalise is the one already recorded
here — which relations you pick has consistently mattered more than what is computed
over them.

### 2. Multi-query KV cache — size confirmed, accuracy says pretrain

`n_train=6000`, `n_features=20`, `nhead=8`:

| | cache | per row |
|---|---:|---:|
| full multi-head | 145.1 MiB | 25362 B |
| single KV head | 18.1 MiB | 3170 B |

Exactly **8.0x**, matching the paper's "factor of eight". Extrapolated to 1M rows:
25.4 GB -> 3.2 GB (TabPFN-3 reports ~7 GB; TabICLv2's ICL dim is smaller).

**But the accuracy result is decisive: ROC-AUC 0.9885 -> 0.3196** — worse than
chance. Averaging 8 trained heads into 1 destroys the model. This is not a tuning
problem; the checkpoint's heads are not redundant. **MQA cannot be retrofitted. It
requires pretraining with `nhead_kv=1`.** The code here is correct and
training-ready, and the measurement is what establishes that the shortcut fails.

### 3. Relational flattening — works, no model change

Synthetic 2-table schema, signal placed only in the child table:

| features | ROC-AUC |
|---|---:|
| entity table only (2 cols) | 0.4946 |
| flattened relational (13 cols) | 0.8535 |

Baseline near chance confirms the lift comes from the aggregation, not leakage. The
cutoff test proves post-cutoff child rows are excluded.

### 4. Test-time compute — real but small

`make_classification`, 1400 x 20:

| | log-loss | ROC-AUC | time |
|---|---:|---:|---:|
| base | 0.2642 | 0.9518 | 3.8 s |
| thinking, n=2 | 0.2596 | 0.9524 | 14.0 s |
| thinking, n=4 | 0.2582 | 0.9527 | 17.7 s |

Monotone improvement, but ~4.6x the compute for ~2% relative log-loss. Nowhere near
TabPFN-3-Plus's claimed +200 Elo — unsurprising, since their mechanism is
undisclosed and this is a generic wrapper rather than a port. Treat this as a
baseline for test-time scaling on TabICL, not a reproduction.

## Enabling row chunking

Row chunking is a first-class `InferenceConfig` option, alongside `offload`:

```python
# always chunk
TabICLClassifier(inference_config={"COL_CONFIG": {"row_chunk": True}})

# chunk only when the projected activation would not fit comfortably
TabICLClassifier(inference_config={"COL_CONFIG": {"row_chunk": "auto"}})
```

`"auto"` decides per call from the real tensor shape and the real free VRAM, so the
same setting adapts to a 12 GB card and an 80 GB one without the caller knowing the
~150k-row threshold. It projects the unchunked peak as `input x 8.6` (measured: a
3277 MiB input peaked at 28140 MiB on an L40S) and chunks when that exceeds
`auto_row_chunk_threshold` (default 0.35) of free memory. On CPU it never engages,
since there is nothing to run out of.

Defaults to off, so existing behaviour is unchanged. It composes with `offload`
and with `kv_cache`: offloading moves *outputs* off the GPU, chunking shrinks the
*activations*, and the two are independent.

`tabicl.scaling.row_chunked(model)` remains as a context manager for ad-hoc use on
an already-fitted estimator.

One wiring note: `InferenceManager.configure` takes an explicit signature with no
`**kwargs`, so the three chunking keys are filtered out by `MgrConfig.manager_items()`
before the call. New caller-side options should follow the same route.

## Equivalence: what "exact" does and does not mean

Chunking is exact at the stage it replaces, but **end-to-end probabilities still move by
~1e-2**, and that is not a bug in the chunking.

Measured on real model tensors (`n_train=3000`, 40 features, fp32, AMP off):

| where | difference |
|---|---|
| `tf_col` output, chunked vs unchunked | **1.16e-05** |
| final `predict_proba`, chunked vs unchunked | 1.2e-02 (0 label flips) |

The control settles it: injecting *random* noise of 1.16e-05 into the column embedding
and changing nothing else moves the output by **1.35e-02** — the same magnitude. So
TabICL's ICL stack (12 attention blocks) amplifies any float32-level perturbation of the
column embedding by ~10^3. Raising the injected noise to 1e-4 gives 1.18e-02, i.e. the
response saturates rather than scaling linearly.

Consequences:

* Assert equivalence at the `tf_col` boundary (~1e-4), not on probabilities.
* End-to-end, compare argmax labels and expect probabilities to agree only to ~2e-2.
* This is a property of the model, not of chunking. Any change to the column embedding
  that is merely float-accurate -- a different kernel, AMP, a new GPU -- will move
  probabilities by the same order.

## Known measurement limits

* The 3060 numbers never observed a true OOM. On Windows, WDDM silently spills to
  host memory, so the 32k-row unchunked run reporting 14078 MiB on a 12288 MiB card
  was paging, not failing — its 15.9 s reflects spill, not compute. On Linux the same
  shape OOMs outright.
* Largest shape tested is 32k rows x 100 features. TabPFN-3's claim concerns 1M
  rows; whether the ~6x holds and whether chunk overhead amortises at 10^5-10^6 rows
  is untested here.
* **Not yet compared against TabICLv2's own `offload` path.** That is the real
  competitive baseline -- the paper's criticism of TabICLv2 is that offloading costs
  ~250 GB host RAM or ~4x slowdown, and chunking is only clearly better if it beats
  offload, not merely the naive path.

### Relation breadth on rel-trial -- nearly a null result

rel-trial has 10 timestamped children and the runner had been using 3, selected by
*row count ascending* -- i.e. cheapest to compute, which has nothing to do with
usefulness. Since "which relations you traverse" had been the largest measured lever
(+0.083), the unused 7 looked like the obvious place to find rel-trial's -8.8 gap,
especially as they include `eligibilities`, `conditions_studies`, `sponsors_studies`
and `facilities_studies` -- plausible predictors of whether a trial succeeds.

At `max_columns=2`, official test split:

| children | features | test ROC-AUC x100 |
|---:|---:|---:|
| 3 (smallest) | 151 | 65.05 |
| 10 (all) | 414 | 65.40 |

**+0.35 for more than tripling the feature count -- which is inside the noise band.**
A five-seed paired comparison on rel-event puts the sd of a paired gap at 0.29, so
anything under about +-0.58 is indistinguishable from zero. The honest statement is that
seven extra relations bought *nothing measurable*, not that they bought a third of a
point. The conclusion is unchanged and if anything stronger: relation breadth is not what
rel-trial is missing, and the arbitrary "smallest first" rule was costing nothing
detectable. Greedy block-level selection over these 10 blocks was written and ready
to run; on this evidence it would be optimising a lever worth a third of a point, so it
is parked rather than run.

The contrast that matters is with categoricals on the same task: the numeric-only as-of
path scores 65.05 where the join path *with mode* scores 67.61. **~2.5 points sit behind
the categorical restriction, against ~0.35 behind relation breadth.** That reorders the
queue below.

## What to try next

Leads worth keeping, roughly in expected-value order. The first three come from reading
a TabICL integration review alongside our own measurements; several of its findings
corroborate results already in this file, and two of them redirect work we had planned.

### 1. Top-K category histograms in `asof_statistics` (replaces the sketch plan)

The open restriction is that the as-of path is numeric-only: no `nunique`, no `mode`.
The plan had been HLL / Space-Saving sketches. There is a better answer.

That review describes a *cluster histogram* for text in child tables: embed the child
text, fit one global k-means codebook, then describe each parent by which clusters its
children fall into and in what proportions -- keeping the distribution over concepts
rather than mean-pooling embeddings into a centroid that corresponds to nothing.

Strip the embedding out and the same construction handles plain categoricals. Choose
the K most frequent categories once, globally; emit K indicator columns; prefix-sum
each one. That is all it takes, because **an indicator is a counter**, and counters are
exactly what the scan already carries. Consequences:

* it needs no new data structure -- no HLL, no Space-Saving, no sketch of any kind
* it is **exact**, not approximate
* counts are invertible, so windows remain prefix differences and stay nearly free
* it strictly dominates `mode`, which is just the argmax of this histogram
* an "other" bucket absorbs the tail, and a raw child count distinguishes 3-of-5 from
  600-of-1000 (proportions alone throw magnitude away)

Cost is K columns per categorical column, and the column-budget result says to keep K
small -- 4 to 8. True `nunique` would still want a sketch, but it is one approximate
scalar against a whole exact distribution, so it is the weakest of the three and can
wait. Target: rel-trial's ~2.5-point categorical gap.

### 2. Sweep the context size before spending more on memory

The same review reports that quality-versus-context curves are **remarkably shallow**
on relational data: H&M churn moves 0.649 -> 0.673 for a *200x* larger context (500 vs
100k rows), and their conclusion is that 10k-20k context rows reach within epsilon of
the best observed quality on every relational dataset they tested.

If that holds here it matters directly. rel-avito used 116,598 context rows and needed
an L40S; at 10-20k it would run on a 12 GB card. It also reframes row chunking as one
cost lever among two, where the other -- use fewer rows -- is free and untested by us.
Cheap experiment: rel-avito at 10k / 20k / 40k context against the measured 64.46.

### 3. The other feature-budget axis

`max_columns` caps *source columns* and keeps every statistic. That review's DFS caveat
proposes the orthogonal cut: keep every column and cap the *aggregates per column*
("drop aggregate features of the same column and keep only 1"). Both shrink the matrix
along different axes and should compose. Given that cap=1 beat cap=4 here, and that
their own DFS feature counts run 14-36 with free text and high-cardinality categoricals
excluded, the combination is worth one sweep.

### 4. GFS, if depth is ever revisited

GFS (*Graph-based Feature Synthesis*, VLDB Workshop 2024) is described as a strict
generalization of DFS: every feature DFS produces, GFS produces, plus more, by covering
*all* join paths up to k hops instead of one traversal -- and it beats DFS specifically
on deeper schemas. Our depth-2 result was a single traversal on one task, so it does not
rule this out. Low priority until something suggests depth pays at all.

### Deliberately not doing

* **Retrieval / nearest-neighbour context selection.** That review tested an
  entity-overlap strategy across four temporal RDBs and it disappointed on all but one
  (Favorita, a pure forecasting task). That is essentially `select_context(method="knn")`.
  Independent evidence against a direction we had not found a reason to pursue either.
* **Greedy block selection over relations.** Parked, per the rel-trial result above.
* **Sketches for as-of categoricals.** Superseded by item 1, except for true `nunique`.

## Auditing three wired interventions before measuring them (2026-08-10)

`RESEARCH.md`'s model-selection-under-shift review named four things to run in order of
expected value per pod-hour, and the three cheapest were wired in three commits the same
evening. None had been run. Reading them before spending a pod hour found that **two of the
three would not have measured what they claimed, and the third would have misreported a
partial failure as a result.** All three failures share a shape: the run completes, prints a
plausible number, and raises nothing.

### `--select-lexi` was inert, and its unit tests all passed

`lexi_select` takes the sweep's candidates plus `split_val`, the per-fold validation scores,
and breaks ties among near-optimal configurations by best worst fold. It was correct, and it
had four passing tests.

`split_val` was only ever populated inside a branch gated on `bool(args.abstain)`. Run with
`--select-lexi` alone, `lexi_select` found the dict empty, took its documented fallback to
the plain argmax, and the run printed a lexicographic selection line while selecting exactly
what it always had. The A/B would have come back **+0.00 with zero variance** — which on this
benchmark means a broken experiment, not a negative result.

**The generalisable point is about where the tests were.** The function had complete unit
coverage; nothing tested that the runner ever handed it data. A unit test on a mechanism does
not test that the mechanism is reachable, and the reachable-ness is what was broken. The
gating is now a module-level `fold_plan` with its own tests, and the abstention *veto* is
gated on `args.abstain` separately, so enabling the folds for lexi cannot smuggle a second
intervention into the arm.

Worth recording that enabling the folds does **not** change what the argmax listens to: the
fold branch takes `return_probs=True` and computes the full-validation AUC from the same
probabilities the ordinary path scores, so the two are numerically identical and the
comparison stays single-variable. That was checked rather than assumed.

### `--random-val-split` drew one slice for the whole run

The draw sat outside the seed loop, from a hardcoded `default_rng(12345)`. Every replicate
therefore selected on the *same* held-out rows: the spread across seeds covered context draws
and model randomness and **excluded the split itself**, which is the one source of variance
the claim is about. arXiv 2502.20260's claim is that *a random split* selects better than a
temporal one — a statement about the family of draws — and a single fixed draw cannot address
it at any replicate count. A lucky slice would have been indistinguishable from a working
method, and the tight spread would have read as confirmation.

Now `random_val_indices(n_train, n_val, seed)`, module-level and redrawn per replicate.

A third defect fell out of fixing the first two: `va_early`/`va_late` were built by ordering
the official `val` table, but the random slice has length `min(len(val), n_train // 3)`, so
combining the two flags indexed one frame's positions into another's. The folds are now built
from the rows actually validated on, ordered by their own train timestamps.

**What is not a confound, having been checked:** the treatment removes the held-out rows from
the *selection* context pool, which is necessary or the same rows would be fitted and scored.
The final test fit calls `draw()` without a pool and therefore draws from the full training
set in both arms. Only what selection listened to differs.

### The drift-resilient backbone arm would have misreported partial failure

Two problems, neither fatal on its own. It searched a candidate list for the *class* — the
right instinct, and the reason it exists — and then **guessed the constructor**, passing
`device=` and `random_state=`. That is the same mistake one level down from the one that cost
a cycle on TabFM, where the loader name was right by luck and its device argument was not.

More consequential: the per-seed failure path records `nan` and continues, which is correct,
but the summary then used `nanmean` over the survivors while dividing the standard error and
the sign count by the number of *attempts*. An arm that raised on eight of twelve seeds would
have printed a confident mean, an SE with the wrong denominator, and "3/12 positive" from
four comparisons. Partial failure is not random here — the seeds that fail are the ones with
awkward context draws — so the surviving subset is biased. It now reports `n=ok/total`, flags
`PARTIAL`, and pairs only seeds where both arms produced a number.

### The pod harness, promoted out of the scratchpad

Every previous version of the round driver was written into a session scratchpad, fixed
there, and lost with the session — the notes record four consecutive rounds that shipped
nothing, and this session rediscovered three more distinct failures before a round started:
a CRLF `_pod_setup.sh` (bash read `set -euo pipefail\r` and reported "invalid option name",
which reads like the wrong shell rather than the wrong file), a payload missing `LICENSE`
which `pyproject` declares, and `pip ... | tail -3` truncating its own diagnosis.

It now lives at `scaling/cycle.py` with `_pod_remote.sh`, and each failure is pinned by a
test. Two more were found by running it:

* **Logs were fetched on the happy path**, so the setup failure that actually occurred raised
  straight past the fetch and terminated the host with `setup.log` still on it. The run that
  most needed its log was the one guaranteed to lose it. Fetching is now in the `finally`.
* **The poller read a dead host as a finished round.** A community L40S was reclaimed nine
  minutes into a five-hour round; `ssh` printed "Connection refused"; that text contains none
  of the probe's three `---` separators, so `partition` put all of it in the `done` slot and
  the poller announced `WORK_DONE`. `parse_probe` now requires a zero return code and the
  separators, and three unanswered polls end the round as **lost, not done**. `TABICL_CLOUD`
  lets a long round demand SECURE, which is a better bet than community capacity for
  something measured in hours.

## Status

| # | Feature | Status |
|---|---|---|
| 1 | Row chunking | **Working.** ~6x less peak memory, exact outputs. |
| 2 | MQA KV cache | **Blocked on pretraining.** 8x size win confirmed; post-hoc collapse unusable. |
| 3 | Relational | **Working.** No model change. |
| 4 | Test-time compute | **Working, modest.** +0.006 log-loss for ~4.6x compute. |
