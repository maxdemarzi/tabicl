# tabicl.scaling — current state

Where this branch stands today. `DESIGN.md` is the history log: derivations, what was
tried, what failed, and why. `PERFORMANCE.md` is the dated log of every measurement and what changed between runs.
`TODO.md` is the open work and `RESEARCH.md` the candidate
directions, scored by what has since been measured. This file is only the present tense.

## What the package provides

Four scaling techniques from *TabPFN-3: Technical Report* (arXiv 2605.13986), plus a
relational feature layer and a compiled join engine underneath it.

| # | Feature | State | Default |
|---|---|---|---|
| 1 | Row-chunked column embedding | Working. Exact, not approximate. | off |
| 2 | Multi-query KV cache | Size win confirmed; needs pretraining to use for accuracy. | off |
| 3 | Relational flattening | Working. No model change. | n/a |
| 4 | Test-time compute | Working, modest. | off |

## Public API

```python
from tabicl.scaling import (
    # 1. row chunking
    row_chunked, chunked_set_transformer, DEFAULT_CHUNK_SIZE, DEFAULT_COL_CHUNK_SIZE,
    # 2. KV cache
    kv_cache_bytes, collapse_kv_heads, expand_kv_heads,
    # 3. relational
    Table, flatten_relational, asof_statistics, hop_product,
    # semirings / FAQ
    Semiring, SUM_PRODUCT, MIN_PLUS, MAX_PLUS, BOOLEAN, BUILTIN_SEMIRINGS,
    check_semiring_laws,
    # joins and graph features
    Atom, wcoj_join, wcoj_count, wcoj_aggregate, native_available,
    triangle_counts, motif_features,
    typed_triangle_counts, typed_motif_features,
    temporal_motif_features, typed_temporal_motif_features,
    # context selection
    select_context, prune_features,
    # graph and label-derived features
    label_homophily, select_graph_context,
    neighbour_label_features, key_target_history,
    # 4. test-time compute
    think_predict_proba, ThinkingResult,
)
```

Also, not exported at the top level but part of the measurement discipline:
`_leakage.permutation_control`, `_leakage.permutation_test`, `_leakage.temporal_control`,
`_calibrate.calibrate_context_size`, `_calibrate.sweep_configurations`.

## Configuration

Row chunking is an `InferenceConfig` option:

```python
TabICLClassifier(inference_config={"COL_CONFIG": {"row_chunk": True}})    # always
TabICLClassifier(inference_config={"COL_CONFIG": {"row_chunk": "auto"}})  # when it won't fit
```

`"auto"` decides per call from the real tensor shape and real free VRAM, projecting the
unchunked peak as `input x 8.6` and chunking above `auto_row_chunk_threshold` (0.35 of
free memory). It never engages on CPU. It composes with `offload`: offloading moves
*outputs* off the GPU, chunking shrinks *activations*, and the two are independent.

`row_chunked(model)` is the equivalent context manager for an already-fitted estimator.

### Calibration

Every lever here is task-dependent, several by margins larger than any average effect,
so the settings are chosen per dataset rather than defaulted:

```python
from tabicl.scaling import calibrate_context_size, sweep_configurations

result = calibrate_context_size(X_tr, y_tr, X_val, y_val, fit_score,
                                candidates=(1000, 5000, 10000, None), tolerance=0.005)
result.chosen      # cheapest context within tolerance of the best
result.curve       # every (setting, score) — a curve still climbing means sweep wider
```

Selection takes the **cheapest** candidate within `tolerance`, not the argmax: the argmax
chases validation noise and hands the saving back. A genuinely steep curve still selects
the expensive end. Subsampling stratifies by default.

## Relational features

```python
from tabicl.scaling import Table, flatten_relational, asof_statistics

table = Table(
    child_df, foreign_key="user_id", name="visits",
    time_column="ts",                     # enables per-row cutoffs and windows
    windows=[pd.Timedelta(days=30)],      # look-back windows
    max_columns=2,                        # rescue for schemas that cannot fit; see below
    top_k_categories=4,                   # per-category proportions (opt-in)
    min_category_share=0.5,               # skip columns a codebook cannot describe
    include_mode=True,                    # modal value, all-history only
    primary_key="id", children=[...],     # depth-2; requires unique entity keys
)

features = flatten_relational(entity_df, "user_id", [table], cutoff_column="ts")
features = asof_statistics(table, keys, cutoffs)   # O(n log n) scan
```

Two aggregation paths, same `Table` spec:

| | `flatten_relational` | `asof_statistics` |
|---|---|---|
| cost | `\|child\| x rows sharing a key` | `O(n log n)` |
| types | numeric + categorical (nunique, mode) | numeric; categorical when opted in |
| windows | supported | supported, nearly free (prefix differences) |
| depth-2 | yes, unique entity keys only | no |
| guard | `MAX_JOIN_PAIRS = 50M` | n/a |

Aggregation carries factorized sufficient statistics (`count/sum/sumsq/min/max`) and
derives `mean`/`std` at the root, so nested levels compose correctly. Variance
accumulates around a per-column pivot, which holds precision at large offsets.

`max_columns` selects by non-null coverage — target-free, so it cannot leak, and
deterministic on ties.

**It is a rescue, not a default.** The same setting is worth +3.0 on one task, 0.0 on
another, and −19.5 on a third (table below). Reach for it when a schema cannot otherwise
run; do not tighten it on one that already fits, and choose the value on a validation
split rather than assuming one.

Categorical statistics on the as-of path are opt-in via `top_k_categories` and
`include_mode`. Both are **off by default**, and the reason is now understood: the effect
is *ensemble-dependent*. On rel-event the blocks are **+2.87 at `n_estimators=1` and
−3.52 at 4**, crossing over between 1 and 2. They help a single estimator and hurt an
ensemble, because a wider feature set costs ensemble diversity — plain gains six points
from ensembling where categorical gains nothing. Since the deployment default is an
ensemble, off is right. Also +1.25 on rel-trial, 0.0 on rel-f1, at 2.3× the model time. The block's price is paid in columns, so it loses on any task
where column count is already the binding constraint. Turn it on when a schema's signal
is genuinely categorical, and verify on a validation split.

Mechanically: each categorical column becomes per-category proportions over a globally
fixed codebook plus an `other` bucket — exact, no sketches, and valid over windows
because counts are invertible. `min_category_share` skips columns whose codebook would
capture too little mass; without that gate the block emits near-constant columns for
free text and hurts badly. `include_mode` adds the modal value, all-history only.

## Relation to Deep Feature Synthesis

The relational layer is DFS-shaped, and saying so is the honest framing: Deep Feature
Synthesis (Kanter & Veeramachaneni, IEEE DSAA 2015; implemented in Featuretools) is the
standard automated approach to exactly this problem — apply aggregation primitives across
foreign-key paths, recursively, to some depth. `flatten_relational` does the same thing.
What differs:

| | DFS / Featuretools | here |
|---|---|---|
| aggregation | primitives applied per level (`COUNT`, `MEAN`, `STD`, `MODE`, `NUM_UNIQUE`, …) | factorized sufficient statistics (`count/sum/sumsq/min/max`), `mean`/`std` derived at the root |
| nesting | stacked primitives, e.g. `MEAN(sessions.SUM(txns.amount))` | statistics compose, so a depth-2 mean is the **true** mean, not a mean of means |
| time | `cutoff_time` per row, `training_window` | per-row cutoff as an `O(n log n)` prefix scan; windows as prefix differences, nearly free |
| algebra | fixed primitive set | semirings — `SUM_PRODUCT`, `MIN_PLUS`, `MAX_PLUS`, `BOOLEAN` — one scan, laws checkable |
| joins | tree-shaped foreign-key paths | plus a compiled WCOJ for **cyclic** patterns (triangles, typed motifs) |
| width | feature count grows sharply with depth | explicit `max_columns` budget, target-free so it cannot leak |
| the target | **never used** | `key_target_history` uses *other rows'* labels, with a resolution horizon and negative controls |

**The last row is the one that matters.** DFS deliberately never touches the target, and
that is precisely the ceiling: on rel-trial, admitting other rows' outcomes was worth
**+5.45**, more than every structural lever in this package combined. The rows above it
are engineering — exactness, cost, cyclic patterns; the row below the line is the only
thing that has moved a headline number.

The mean-of-means distinction is a real correctness difference, not a preference. Stacked
primitives compute a mean over groups of a per-group statistic, which does not equal the
overall statistic when group sizes vary. Sufficient statistics compose to the exact
root-level quantity. Both are legitimate features; only one is the number you probably
meant.

**Now benchmarked against DFS.** Featuretools was run over the same four tasks with
per-row cutoff times and scored by the same TabICL, same context, same seeds — only the
feature builder differs, which makes it the one comparison isolating *our aggregation*
from *our model*. DFS was given the tuned primitive set, not the default one.

Paired by seed, so the "ours" column is our layer *in the matching configuration* and not
the calibrated headline number:

| task | DFS | ours | ours − DFS | feature build |
|---|---:|---:|---:|---|
| rel-f1 / driver-top3 | 76.81 | 81.84 | **+5.03** (sd 1.49, 5/5) | 28 s → **1 s** |
| rel-event / user-ignore | 77.95 | 80.34 | **+2.39** (sd 3.16, 4/5) | 119 s → **25 s** |
| rel-trial / study-outcome | 69.12 | 69.56 | +0.45 (sd 0.50, 5/5) | 20 s → **2 s** |
| rel-avito / user-visits | 65.81 | 65.51 | −0.30 (sd 0.52, 2/5) | 414 s → **18 s** |

Two wins, two ties inside the ±0.6 floor, and features built 5–23× faster — on rel-avito
DFS spent seven minutes to produce 94 columns. So "a generic flattening pipeline" was too
modest on half the benchmark and exactly right on the other half.

Two caveats worth carrying. Both sides used three child tables, so this is DFS-on-3
against ours-on-3 rather than against the full layer. And **rel-avito is the one task
where DFS is ahead**, which is also the task where we sit nearest a published win — those
two facts probably belong together.

See `PERFORMANCE.md`, 2026-08-05 entry (RESEARCH 6e), for the protocol and per-seed numbers.

## Measured results

### RelBench, official protocol

Fit on `train`+`val`, score the held-out `test` split. Test ROC-AUC x100. Comparison
columns are published figures from the TabPFN-3 paper's Table 14.

All numbers below are **calibrated**: every setting chosen on a validation split, test
touched once, AMP off, selection and scoring at the same `n_estimators`. The earlier
per-task-maximum table selected its configurations *on test*, ran about a point higher,
and was not comparable with the published figures; it has been retired.

| task | ours | TabPFN-REL | RelGNN | RDBLearn+v3 | vs best |
|---|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | 80.70 | 79.98 | **85.69** | 82.72 | −4.99 |
| rel-event / user-ignore | 78.11 | 85.38 | **86.18** | 73.70 | −8.07 |
| rel-avito / user-visits | 64.85 | 66.68 | 66.18 | **66.76** | −1.91 |
| rel-trial / study-outcome | 69.36 | **76.43** | 71.24 | 72.89 | −7.07 |

**We win no task.** Ahead of TabPFN-REL on rel-f1 and behind RelGNN there by 5; within 2
on rel-avito; well behind on rel-event and rel-trial. rel-trial moved 66.50 → 69.36 on
2026-08-04 via the shared-key track record, and is still last on that task — a narrowed
gap is not a win. This is a generic flattening pipeline in front of a stock TabICL, with
no relational machinery in the model and no retraining, measured against systems built for
relational data.

**Measurement floor: ±0.6.** Paired-gap sd is 0.29 on rel-event, absolute-score sd 2.28 —
pairing tightens by ~8x. Differences under ~0.6 are not resolvable, and unpaired
comparisons cannot resolve five points. Pair everything, and see `PERFORMANCE.md` for what
changed between runs.

**On GPU, set `use_amp=False` for any accuracy measurement.** The default is `True` and it
scored 73.61 against CPU's 80.93 on rel-event; `use_amp=False` reproduces CPU exactly. It
is harmless on well-separated data and expensive exactly where the model is uncertain.
`_evalcfg.py` makes fp32 the default for the eval scripts and `--amp` opt-in.

Calibration is better protocol, not a cure: on rel-event it chose the categorical blocks
on a 0.49 validation margin, and those blocks measure −3.21 on test there. Validation
splits of 825–2,013 rows are small enough that selection noise is real.

### What moves the number

Ranked by measured AUC contribution, largest first:

| lever | effect |
|---|---|
| **AMP on/off** | **−7.3 on rel-event** — the largest single effect found; `use_amp=False` |
| **shared-key track record** | **+5.45 on rel-trial** (sd 0.30, 5/5 seeds); enters the headline |
| column budget | +3.0 rel-event, 0.0 rel-trial, **−19.5 rel-f1** — task-dependent |
| graph-neighbour context | +3.10 on rel-event, but validation rejects it 4/5 — unusable |
| categorical blocks (as-of) | ensemble-dependent: **+2.87 at n_estimators=1, −3.52 at 4** — off by default |
| context size | rel-avito loses nothing at 8.6%; rel-trial loses 2.74 at 23% — calibrate |
| neighbour-label features (graph) | −0.74 on rel-event; real but already captured |
| type-aware motifs | +0.021 |
| which relations you traverse | +0.083 |
| look-back windows | +0.089 on rel-f1; median +0.001 elsewhere |
| the whole WCOJ/FAQ engine | +0.016 |

Choosing *what data enters the features* has consistently beaten *what is computed
over them* — and the largest genuine gain so far came from admitting a related row's
**outcome**, which flattening cannot reach at all.

### Label-derived features

`key_target_history` and `neighbour_label_features` build features from *other rows'
labels* — the one thing flattening structurally cannot do, since it aggregates a related
row's columns and never its outcome. Both are dangerous: a mistake returns a large
confident number rather than an exception. Three rules, each learned the hard way:

1. **Apply a resolution horizon.** A label is not knowable at its own prediction time;
   RelBench's `task.timedelta` is 7 days on rel-event and **365 on rel-trial**. Omitting
   it inflated a rel-event measurement by 6.4 AUC.
2. **Gate before building.** Measure each key's standalone AUC and coverage first
   (`eval_track_record --gate`, no GPU). A key worth less than the pipeline it must
   improve has no room. That ratio predicted rel-trial working and the graph version not.
3. **Give counts their own arm.** A "label" feature's *count* often carries the signal:
   on rel-event `labelled_degree` alone scored 73.2 while the rate scored 67.8. Without a
   counts-only arm the two are indistinguishable.

Controls live in `_leakage.py`. Use `permutation_control` when a feature's whole content
is labels, and `permutation_test` when it also encodes structure — judging a graph feature
against chance calls a sound feature a leak, because degree survives permutation.

### Memory

Row chunking, column embedder, end-to-end `predict_proba` on an L40S (100 features):

| n_train | default | row-chunked |
|---:|---:|---:|
| 60,000 | 22,376 MiB | **15,827 MiB** |
| 150,000 | 31,257 MiB | **24,900 MiB** |
| 300,000 | 38,569 MiB | 38,082 MiB |

Up to ~1.4x below ~300k rows; above that the peak has moved elsewhere. Combine with
`offload="cpu"` at scale.

MQA KV cache is 8x smaller at `nhead=8`, exactly as the arithmetic predicts.

Test-time compute: +0.006 log-loss for ~4.6x compute.

## Compiled backend

`_wcoj_native.cpp` implements the Umbra hash worst-case-optimal join (Freitag et al.,
VLDB 2020, Algorithm 3): iterate the smallest candidate set, probe the others. Product
cost model, cost-descending dynamic scheduling, recursive pre-splitting. Parallel.

Build with `python -m tabicl.scaling.build_native`. Check with `native_available()`;
everything falls back to the Python path when it is absent. On Windows this needs MSVC
Build Tools — clang alone cannot link a CPython extension.

## Current limits

- **Row chunking cannot bound a wide table.** The activation is allocated at
  `embedding.py:444` (`src[..., :train_size, :] + y_emb`), before `tf_col` is reached,
  so wrapping `tf_col` cannot help. Feature-count problems need `max_columns`; row-count
  problems are what chunking is for.
- **Depth-2 requires unique entity keys.** Available on rel-trial; not on rel-f1 or
  rel-event, whose task tables repeat keys.
- **`asof_statistics` covers categoricals only partly.** `nunique` and `mode` are
  all-history only; over a *window* `nunique` needs an offline dominance count and
  `mode` has no range algorithm. Per-category proportions do work over windows. No
  depth-2 on this path.
- **MQA cannot be applied post hoc.** Collapsing a trained model's heads destroys
  accuracy; the 8x size win requires pretraining with MQA.
- **ICL-stage chunking is off by default.** `InferenceManager` already batches that
  stage, so it changes nothing end to end. Kept for callers that bypass the manager.
- **Windows are only supported with a cutoff column.** Without one there is no
  reference point, and the call is rejected rather than silently ignoring them.
- **Graph-neighbour context is not selectable.** It measures +3.10 to +4.94 over a random
  context of equal size on rel-event and reproduces across seeds, hops and stratification
  — but the calibrated protocol rejects it 4 times in 5, and validation rates it ~9 points
  *below* random while test rates it above. Not a split difference: the standalone
  neighbour signal is 73.9 val / 74.2 test. Until a selection signal exists that picks it,
  it cannot be reported. See `RESEARCH.md` item 8.
- **`key_target_history` counts an entity once per shared key**, so its rate is a
  membership-weighted average rather than a distinct-entity one. Multi-hop label
  propagation is not implemented: on rel-event one hop already reaches 7,111 of 8,517
  eligible users, so two hops is the whole component and the feature goes constant.

## Tests

`tests/test_scaling.py` — 168 passed, 1 skipped. The skip is the compiled backend when
it has not been built.

## Running measurements

**Never locally.** `pod_runner.py` drives a RunPod host end to end, and gates it on both
CUDA-from-Python *and* download bandwidth (5 MB/s floor) before scheduling work — one host
had a healthy `nvidia-smi` and 270 kB/s, which cost a whole session. Every exit path stops
the pod unless `--keep` is passed.

```
python -m tabicl.scaling.pod_runner create      # rejects unusable hosts, up to 8 attempts
python -m tabicl.scaling.pod_runner terminate   # bills by the second; do not leave it
```

Before trusting any number from a new host, reproduce a recorded cell **with the runner
that recorded it** — a near-miss from a different runner proves nothing, since feature
builds differ between them.
