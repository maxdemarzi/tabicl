"""Scaling upgrades ported from TabPFN-3 onto TabICLv2.

See ``DESIGN.md`` for scope, derivations, and the one technique (multi-query
attention) that cannot be fully realised without pretraining.
"""

from ._mqa import collapse_kv_heads, expand_kv_heads, kv_cache_bytes
from ._relational import Table, asof_statistics, flatten_relational, hop_product
from ._calibrate import Calibration, calibrate_context_size, sweep_configurations
from ._leakage import LeakageReport, permutation_control, temporal_control
from ._graph_context import label_homophily, select_graph_context
from ._retrieval import select_context
from ._selection import prune_features
from ._semiring import (
    BOOLEAN,
    BUILTIN_SEMIRINGS,
    MAX_PLUS,
    MIN_PLUS,
    SUM_PRODUCT,
    Semiring,
    check_semiring_laws,
)
from ._rowchunk import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COL_CHUNK_SIZE,
    chunked_set_transformer,
    row_chunked,
)
from ._ttc import ThinkingResult, think_predict_proba
from ._wcoj import (
    Atom,
    motif_features,
    native_available,
    temporal_motif_features,
    triangle_counts,
    typed_motif_features,
    typed_temporal_motif_features,
    typed_triangle_counts,
    wcoj_aggregate,
    wcoj_count,
    wcoj_join,
)

__all__ = [
    "chunked_set_transformer",
    "row_chunked",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COL_CHUNK_SIZE",
    "kv_cache_bytes",
    "collapse_kv_heads",
    "expand_kv_heads",
    "Table",
    "flatten_relational",
    "hop_product",
    "asof_statistics",
    "Semiring",
    "SUM_PRODUCT",
    "MIN_PLUS",
    "MAX_PLUS",
    "BOOLEAN",
    "BUILTIN_SEMIRINGS",
    "check_semiring_laws",
    "label_homophily",
    "select_graph_context",
    "LeakageReport",
    "permutation_control",
    "temporal_control",
    "Calibration",
    "calibrate_context_size",
    "sweep_configurations",
    "select_context",
    "prune_features",
    "think_predict_proba",
    "ThinkingResult",
    "Atom",
    "wcoj_join",
    "wcoj_count",
    "wcoj_aggregate",
    "triangle_counts",
    "motif_features",
    "typed_triangle_counts",
    "typed_motif_features",
    "temporal_motif_features",
    "typed_temporal_motif_features",
    "native_available",
]
