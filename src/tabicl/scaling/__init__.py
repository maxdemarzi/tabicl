"""Scaling upgrades ported from TabPFN-3 onto TabICLv2.

See ``DESIGN.md`` for scope, derivations, and the one technique (multi-query
attention) that cannot be fully realised without pretraining.
"""

from ._mqa import collapse_kv_heads, expand_kv_heads, kv_cache_bytes
from ._relational import Table, flatten_relational
from ._rowchunk import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COL_CHUNK_SIZE,
    chunked_set_transformer,
    row_chunked,
)
from ._ttc import ThinkingResult, think_predict_proba
from ._wcoj import Atom, motif_features, native_available, triangle_counts, wcoj_join

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
    "think_predict_proba",
    "ThinkingResult",
    "Atom",
    "wcoj_join",
    "triangle_counts",
    "motif_features",
    "native_available",
]
