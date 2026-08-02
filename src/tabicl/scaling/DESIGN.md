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

**Scope.** Depth-1..k aggregation over foreign keys (count/mean/sum/min/max/nunique
for numerics, mode/nunique for categoricals), with timestamp truncation so no
child row after the cutoff leaks in. Pure pandas.

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
TabICLClassifier(
    inference_config={"COL_CONFIG": {"row_chunk": True, "row_chunk_size": 8192, "col_chunk_size": 32}}
)
```

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
