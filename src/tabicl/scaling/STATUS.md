# tabicl.scaling — current state

Where this branch stands today. `DESIGN.md` is the history log: derivations, what was
tried, what failed, and why. `TODO.md` is the open work, including one unresolved
question that a few claims below depend on, and `RESEARCH.md` the candidate directions
after it. This file is only the present tense.

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
    # 4. test-time compute
    think_predict_proba, ThinkingResult,
)
```

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
| cost | `|child| x rows sharing a key` | `O(n log n)` |
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

## Measured results

### RelBench, official protocol

Fit on `train`+`val`, score the held-out `test` split. Test ROC-AUC x100. Comparison
columns are published figures from the TabPFN-3 paper's Table 14.

> **Read this table as a per-task maximum, not as a procedure.** Each row is the best of
> several configurations — `max_columns` on rel-event, join-versus-scan on rel-trial,
> context size on rel-avito — and those configurations were compared *on the test split*.
> That is selection on test, and it inflates the column by an unknown amount. The
> published figures it sits beside are presumably single-configuration results, so the
> comparison is not like-for-like in our favour.
>
> `python -m tabicl.scaling.eval_relbench_calibrated <dataset> <task>` runs the honest
> version: every setting chosen on a validation split, test touched once. Expect lower
> numbers.

| task | calibrated | hand-picked | TabPFN-REL | RelGNN | RDBLearn+v3 |
|---|---:|---:|---:|---:|---:|
| rel-f1 / driver-top3 | **80.70** | 79.77 | 79.98 | 85.69 | 82.72 |
| rel-event / user-ignore | 78.11 ‡ | 80.81 | 85.38 | 86.18 | 73.70 |
| rel-avito / user-visits | *not run* | 64.46 † | 66.68 | 66.18 | 66.76 |
| rel-trial / study-outcome | 66.50 | 67.61 | 76.43 | 71.24 | 72.89 |

‡ Selected at `n_estimators=1` and scored at 4, which is unsound now that the sign is
known to flip with ensemble size. Needs re-running with the fixed default.

† Measured on GPU with the default `use_amp=True`, which costs 7.3 AUC on rel-event.
Likely understated; needs re-measuring with `use_amp=False`.

**On GPU, set `use_amp=False` for any accuracy measurement.** The default is `True` and it
scored 73.61 against CPU's 80.93 on rel-event; `use_amp=False` reproduces CPU exactly. It
is harmless on well-separated data and expensive exactly where the model is uncertain.

**Quote the calibrated column.** It is the one produced by a single procedure with every
setting chosen on validation. The hand-picked column selected its configuration on the
test split, which is worth about a point of inflation (mean 76.06 vs 75.10 over the three
tasks measured so far) and is not comparable with the published figures beside it.

Calibration is better protocol, not a cure: on rel-event it chose the categorical blocks
on a 0.49 validation margin, and those blocks measure −3.21 on test there. Validation
splits of 960–2,013 rows are small enough that selection noise is real.

Level with TabPFN-REL on rel-f1, −2.2 on rel-avito, −4.6 on rel-event, −8.8 on
rel-trial. This is a generic flattening pipeline in front of a stock TabICL, against
systems built for relational data.

### What moves the number

Ranked by measured AUC contribution, largest first:

| lever | effect |
|---|---|
| which relations you traverse | +0.083 |
| look-back windows | +0.089 on rel-f1; median +0.001 elsewhere |
| column budget | +3.0 rel-event, 0.0 rel-trial, **−19.5 rel-f1** — task-dependent |
| categorical blocks (as-of) | ensemble-dependent: **+2.87 at n_estimators=1, −3.52 at 4** — off by default |
| context size | rel-avito loses nothing at 8.6%; rel-trial loses 2.74 at 23% — calibrate |
| type-aware motifs | +0.021 |
| the whole WCOJ/FAQ engine | +0.016 |

Choosing *what data enters the features* has consistently beaten *what is computed
over them*.

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

## Tests

`tests/test_scaling.py` — 132 passed, 1 skipped. The skip is the compiled backend when
it has not been built.
