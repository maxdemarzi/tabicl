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
from typing import Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ._semiring import MAX_PLUS, MIN_PLUS, SUM_PRODUCT, Semiring

__all__ = ["Table", "flatten_relational", "hop_product"]

# Suffix -> the semiring whose `add` rolls that statistic up one level. Naming the
# algebra rather than the pandas function keeps the invertibility fact attached: the
# SUM_PRODUCT ones can be retracted when a row is deleted, the tropical ones cannot.
_STAT_SEMIRING = {
    "count": SUM_PRODUCT,
    "sum": SUM_PRODUCT,
    "sumsq": SUM_PRODUCT,
    "min": MIN_PLUS,
    "max": MAX_PLUS,
}
_STAT_ROLLUP = {suffix: ring.pandas_agg for suffix, ring in _STAT_SEMIRING.items()}


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


def _resolve(child: Table, cutoff_by_key: Optional[pd.Series]) -> Tuple[pd.DataFrame, Set[str]]:
    """Fold ``child.children`` into ``child.df``, returning it plus its stat columns.

    ``cutoff_by_key`` maps this table's foreign key -> the entity's cutoff, so the
    deadline propagates down the chain: a grandchild recorded after the entity's
    prediction time is still leakage, however many hops away it sits.
    """
    df = child.df
    if not child.children:
        return df, set()

    if child.primary_key is None:
        raise ValueError(f"table {child.name!r} has children, so it needs primary_key")

    # Each grandchild is filtered against the cutoff its *parent row* inherits.
    if cutoff_by_key is not None:
        parent_cutoff = df[child.foreign_key].map(cutoff_by_key)
        parent_cutoff.index = df[child.primary_key].values
    else:
        parent_cutoff = None

    blocks = [df.reset_index(drop=True)]
    stat_cols: Set[str] = set()
    for grandchild in child.children:
        block = _aggregate(grandchild, df[child.primary_key], parent_cutoff)
        stat_cols |= set(block.columns)
        blocks.append(block)
    return pd.concat(blocks, axis=1), stat_cols


def _aggregate(child: Table, keys: pd.Series, cutoff: Optional[pd.Series]) -> pd.DataFrame:
    """Aggregate one child table to one row per key, as sufficient statistics.

    Emits only decomposable aggregates (count/sum/sumsq/min/max). Anything a caller
    actually wants -- mean, std -- is derived once at the root by
    :func:`_derive_features`. See that function for why.
    """
    df, nested_stats = _resolve(
        child, pd.Series(cutoff.values, index=keys.values) if cutoff is not None else None
    )

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
    index = pd.Index(keys.values, name=child.foreign_key)
    out = pd.DataFrame(index=index)
    out[f"{stem}__count"] = grouped.size()

    for col in use_cols:
        combiner = _STAT_ROLLUP.get(col.rsplit("__", 1)[-1]) if col in nested_stats else None
        if combiner is not None:
            # Already a sufficient statistic from a deeper level. Roll it up with the
            # combiner its own suffix names -- summing sums, min-ing mins -- rather
            # than aggregating it as if it were raw data. This is what keeps depth-k
            # linear instead of exponential, and what makes the result equal to the
            # aggregate over the fully joined table.
            out[f"{stem}__{col}"] = grouped[col].agg(combiner)
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out[f"{stem}__{col}__count"] = grouped[col].count()
            out[f"{stem}__{col}__sum"] = grouped[col].sum()
            out[f"{stem}__{col}__sumsq"] = df.assign(_sq=series**2).groupby(child.foreign_key)["_sq"].sum()
            out[f"{stem}__{col}__min"] = grouped[col].min()
            out[f"{stem}__{col}__max"] = grouped[col].max()
        else:
            # nunique and mode are NOT semiring aggregates -- distinct-count cannot be
            # rolled up exactly without a sketch (HyperLogLog et al.), and a mode of
            # modes is not a mode. At depth 1 these are exact; deeper, they become
            # "per-parent distinct count" summarised further, which is a different
            # quantity. Documented rather than silently wrong.
            out[f"{stem}__{col}__nunique"] = grouped[col].nunique()
            out[f"{stem}__{col}__mode"] = grouped[col].agg(
                lambda s: s.value_counts().index[0] if len(s) else np.nan
            )

    out[f"{stem}__count"] = out[f"{stem}__count"].fillna(0)
    return out.reset_index(drop=True)


def _derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn sufficient statistics into the features a model wants.

    mean and std are not decomposable over a join, but they are *functions of*
    statistics that are: mean = sum/count, var = sumsq/count - mean^2. Deriving them
    once here gives the true aggregate over the joined table, whereas aggregating
    means level by level gives a mean of means -- for a user with a 3-item order and a
    1-item order, 51.0 instead of 26.5.

    ``sumsq`` is dropped afterwards; it is a carrier, not a feature.
    """
    derived = {}
    for col in frame.columns:
        if not col.endswith("__sum"):
            continue
        base = col[: -len("__sum")]
        count_col = f"{base}__count"
        if count_col not in frame.columns:
            continue
        count = frame[count_col].replace(0, np.nan)
        mean = frame[col] / count
        derived[f"{base}__mean"] = mean
        sumsq_col = f"{base}__sumsq"
        if sumsq_col in frame.columns:
            var = (frame[sumsq_col] / count) - mean**2
            derived[f"{base}__std"] = np.sqrt(var.clip(lower=0))

    if derived:
        frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    return frame.drop(columns=[c for c in frame.columns if c.endswith("__sumsq")])


def hop_product(
    frame: pd.DataFrame,
    left: str,
    right: str,
    name: str,
    semiring: Semiring = SUM_PRODUCT,
) -> pd.Series:
    """Combine two columns across a hop with the semiring's ``mul``.

    The roll-up in :func:`flatten_relational` only ever applies ``add`` -- it sums
    child sums, mins child mins. Combining *across* a hop is a different operation:
    a per-order discount times that order's item total, a probability along a chain,
    a cost accumulated hop by hop. That is ``mul``, and without it those features
    cannot be expressed at all.

    Parameters
    ----------
    frame : pd.DataFrame
        Flattened output containing both columns.

    left, right : str
        Columns to combine.

    name : str
        Name for the resulting series.

    semiring : Semiring, default=SUM_PRODUCT
        Supplies ``mul``. ``SUM_PRODUCT`` multiplies; ``MIN_PLUS``/``MAX_PLUS`` add
        (a cost accumulates along a path even though alternatives are min-ed).

    Returns
    -------
    pd.Series
        ``mul(frame[left], frame[right])``, with ``semiring.one`` for missing values
        so an absent factor is neutral rather than poisoning the product.

    Examples
    --------
    >>> feat["ord__value"] = hop_product(  # doctest: +SKIP
    ...     feat, "ord__discount__mean", "ord__item__price__sum", "ord__value"
    ... )
    """
    for column in (left, right):
        if column not in frame.columns:
            raise ValueError(f"column {column!r} not in frame")

    lhs = frame[left].fillna(semiring.one)
    rhs = frame[right].fillna(semiring.one)
    if semiring is SUM_PRODUCT:
        out = lhs * rhs
    elif semiring in (MIN_PLUS, MAX_PLUS):
        out = lhs + rhs
    else:
        out = pd.Series([semiring.mul(a, b) for a, b in zip(lhs, rhs)], index=frame.index)
    return out.rename(name)


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

    return _derive_features(pd.concat(parts, axis=1))
