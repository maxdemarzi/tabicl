"""Relational data support by flattening a database into one table.

TabPFN-3 (arXiv 2605.13986, Section 3.4) reaches SOTA-among-foundation-models on
RelBenchV1 without any relational machinery in the model or the prior: TabPFN-REL
follows RDBLearn, which converts a tabular foundation model into a relational one by
*"automatically flattening the underlying database into a table"*.

So this is pure featurisation -- no model change, no retraining. It works with
``TabICLClassifier``/``TabICLRegressor`` unmodified.

Temporal correctness matters more than feature richness here: RelBench truncates each
database at the test timestamp before building context, and aggregating a child row
recorded after the cutoff leaks the future. ``cutoff_column`` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Table", "flatten_relational"]

_NUMERIC_AGGS = ("count", "mean", "sum", "min", "max", "std")
# Depth >= 2 multiplies: one grandchild column becomes len(_NUMERIC_AGGS) child
# features, each of which the parent aggregates again -- 25 columns from one.
# Deeper levels therefore use a reduced set. Pass Table(columns=...) to narrow
# further; TabICL has a practical feature ceiling.
_NESTED_AGGS = ("count", "mean", "sum", "max")


@dataclass
class Table:
    """A child table joined to the entity table by a foreign key.

    Parameters
    ----------
    df : pd.DataFrame
        The child rows.

    foreign_key : str
        Column in ``df`` matching the entity table's primary key.

    name : str
        Prefix for generated column names.

    time_column : Optional[str], default=None
        Timestamp column used for cutoff filtering. Required when the caller
        passes a cutoff.

    columns : Optional[Sequence[str]], default=None
        Restrict aggregation to these columns. ``None`` uses every column except
        the key and the timestamp.

    primary_key : Optional[str], default=None
        This table's own identifier. Required only when it has ``children``, since
        that is what the grandchildren's foreign keys point at.

    children : Sequence[Table], default=()
        Tables one hop further out. They are aggregated into this table first, then
        this table is aggregated into the entity -- so a two-hop schema such as
        ``user -> transaction -> product`` collapses in the right order. Depth is
        arbitrary; each level costs one more groupby.
    """

    df: pd.DataFrame
    foreign_key: str
    name: str
    time_column: Optional[str] = None
    columns: Optional[Sequence[str]] = field(default=None)
    primary_key: Optional[str] = None
    children: Sequence["Table"] = field(default=())


def _resolve(child: Table, cutoff_by_key: Optional[pd.Series]) -> pd.DataFrame:
    """Fold ``child.children`` into ``child.df``, one column block per grandchild.

    ``cutoff_by_key`` maps this table's foreign key -> the entity's cutoff, so the
    deadline propagates down the chain: a grandchild recorded after the entity's
    prediction time is still leakage, however many hops away it sits.
    """
    df = child.df
    if not child.children:
        return df

    if child.primary_key is None:
        raise ValueError(f"table {child.name!r} has children, so it needs primary_key")

    # Each grandchild is filtered against the cutoff its *parent row* inherits.
    if cutoff_by_key is not None:
        parent_cutoff = df[child.foreign_key].map(cutoff_by_key)
        parent_cutoff.index = df[child.primary_key].values
    else:
        parent_cutoff = None

    blocks = [df.reset_index(drop=True)]
    for grandchild in child.children:
        blocks.append(
            _aggregate(grandchild, df[child.primary_key], parent_cutoff)
        )
    return pd.concat(blocks, axis=1)


def _aggregate(child: Table, keys: pd.Series, cutoff: Optional[pd.Series]) -> pd.DataFrame:
    """Aggregate one child table down to one row per key in ``keys``."""
    df = _resolve(child, pd.Series(cutoff.values, index=keys.values) if cutoff is not None else None)

    if cutoff is not None:
        if child.time_column is None:
            raise ValueError(f"table {child.name!r} needs time_column to honour a cutoff")
        # Map each child row to its entity's cutoff, then drop rows at or after it.
        # Dropping unmatched keys is intentional: they contribute no history.
        per_row_cutoff = df[child.foreign_key].map(pd.Series(cutoff.values, index=keys.values))
        df = df[df[child.time_column] < per_row_cutoff]

    use_cols = child.columns
    if use_cols is None:
        # primary_key is an identifier -- aggregating it produces noise that looks
        # like signal, so drop it along with the join key and the timestamp.
        excluded = {child.foreign_key, child.time_column, child.primary_key}
        use_cols = [c for c in df.columns if c not in excluded]

    stem = child.name
    grouped = df.groupby(child.foreign_key)
    out = pd.DataFrame(index=pd.Index(keys.values, name=child.foreign_key))
    out[f"{stem}__count"] = grouped.size()

    for col in use_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            aggs = _NESTED_AGGS if child.children else _NUMERIC_AGGS
            agg = grouped[col].agg([a for a in aggs if a != "count"])
            for a in agg.columns:
                out[f"{stem}__{col}__{a}"] = agg[a]
        else:
            out[f"{stem}__{col}__nunique"] = grouped[col].nunique()
            # mode() per group is quadratic on wide categoricals; first-most-common
            # via value_counts is close enough and far cheaper.
            out[f"{stem}__{col}__mode"] = grouped[col].agg(
                lambda s: s.value_counts().index[0] if len(s) else np.nan
            )

    out[f"{stem}__count"] = out[f"{stem}__count"].fillna(0)
    return out.reset_index(drop=True)


def flatten_relational(
    entity_df: pd.DataFrame,
    primary_key: str,
    children: Sequence[Table],
    cutoff_column: Optional[str] = None,
) -> pd.DataFrame:
    """Flatten an entity table plus its child tables into a single feature table.

    Parameters
    ----------
    entity_df : pd.DataFrame
        One row per prediction entity.

    primary_key : str
        Entity identifier column, referenced by each child's ``foreign_key``.

    children : Sequence[Table]
        Child tables to aggregate.

    cutoff_column : Optional[str], default=None
        Column in ``entity_df`` giving each entity's prediction timestamp. When set,
        child rows at or after that timestamp are excluded -- the temporal
        correctness RelBench requires.

    Returns
    -------
    pd.DataFrame
        ``entity_df`` (minus key and cutoff columns) with aggregated child features
        appended. Row order matches ``entity_df``.

    Notes
    -----
    Children may themselves have children; the deeper level is folded in first, so a
    ``user -> transaction -> product`` schema collapses in the correct order and the
    entity's cutoff propagates all the way down.

    Examples
    --------
    >>> feat = flatten_relational(  # doctest: +SKIP
    ...     users, "user_id",
    ...     [Table(txns, "user_id", "txn", time_column="ts")],
    ...     cutoff_column="predict_at",
    ... )
    """
    if primary_key not in entity_df.columns:
        raise ValueError(f"primary_key {primary_key!r} not in entity_df")

    keys = entity_df[primary_key]
    cutoff = entity_df[cutoff_column] if cutoff_column else None
    if cutoff_column and cutoff_column not in entity_df.columns:
        raise ValueError(f"cutoff_column {cutoff_column!r} not in entity_df")

    parts = [entity_df.drop(columns=[c for c in (primary_key, cutoff_column) if c]).reset_index(drop=True)]
    for child in children:
        parts.append(_aggregate(child, keys, cutoff))

    return pd.concat(parts, axis=1)
