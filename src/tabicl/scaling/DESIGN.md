# Porting TabPFN-3 scaling techniques to TabICLv2

Design + scope for four upgrades derived from *TabPFN-3: Technical Report*
(arXiv 2605.13986, Prior Labs, 2026-05-12).

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

**Guidance:** enable `row_chunk` alone below ~150k rows; combine it with
`offload="cpu"` above that.

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

## Status

| # | Feature | Status |
|---|---|---|
| 1 | Row chunking | **Working.** ~6x less peak memory, exact outputs. |
| 2 | MQA KV cache | **Blocked on pretraining.** 8x size win confirmed; post-hoc collapse unusable. |
| 3 | Relational | **Working.** No model change. |
| 4 | Test-time compute | **Working, modest.** +0.006 log-loss for ~4.6x compute. |
