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
    # The pivot each column is accumulated around. Constant per column, so `max`
    # carries it through a roll-up unchanged; it is consumed and dropped at the root.
    "shift": MAX_PLUS,
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

    windows : Sequence, default=()
        Look-back windows, e.g. ``[pd.Timedelta(days=30), pd.Timedelta(days=90)]``.
        Each emits its own block of statistics over child rows falling in
        ``[cutoff - window, cutoff)``, alongside the all-history block. Recency is
        usually the strongest signal a history carries and an all-time mean dilutes
        it: a driver's lifetime average says little about current form. Requires
        ``time_column`` and a cutoff.

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
    windows: Sequence = field(default=())
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

    if cutoff_by_key is not None:
        parent_cutoff = df[child.foreign_key].map(cutoff_by_key)
        parent_cutoff.index = df[child.primary_key].values
    else:
        parent_cutoff = None

    blocks = [df.reset_index(drop=True)]
    stat_cols: Set[str] = set()
    for grandchild in child.children:
        block = _aggregate_by_key(grandchild, df[child.primary_key], parent_cutoff)
        stat_cols |= set(block.columns)
        blocks.append(block)
    return pd.concat(blocks, axis=1), stat_cols


def _stat_columns(
    child: Table, df: pd.DataFrame, nested_stats: Set[str], grouped, index, group_key: str
) -> pd.DataFrame:
    """Emit sufficient statistics for one grouping, shared by both aggregation paths."""
    use_cols = child.columns
    if use_cols is None:
        excluded = {child.foreign_key, child.time_column, child.primary_key}
        use_cols = [c for c in df.columns if c not in excluded and not c.startswith("__")]

    stem = child.name
    out = pd.DataFrame(index=index)
    out[f"{stem}__count"] = grouped.size()

    for col in use_cols:
        combiner = _STAT_ROLLUP.get(col.rsplit("__", 1)[-1]) if col in nested_stats else None
        if combiner is not None:
            # Already a sufficient statistic from a deeper level: roll it up with the
            # combiner its own suffix names -- summing sums, min-ing mins -- rather
            # than aggregating it as if it were raw data. This is what keeps depth-k
            # linear instead of exponential.
            out[f"{stem}__{col}"] = grouped[col].agg(combiner)
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # Accumulate around a pivot. sumsq of raw values, then E[X^2] - E[X]^2,
            # loses every significant digit on large-magnitude columns: measured
            # returning std 18.5 for a true value of 1.0 on data offset by 1e9.
            # Centring keeps the running sums near zero where doubles have precision
            # to spare. Both centred accumulators are still additive, so multi-hop
            # roll-up is unaffected; the pivot rides along and is undone at the root.
            #
            # Vectorised, not `.apply(lambda ...)`: a Python callback per group was
            # measured at a third of total runtime for no gain in accuracy.
            values = series.astype("float64")
            shift = float(values.mean()) if values.notna().any() else 0.0
            centred = df.assign(**{"__c": values - shift, "__csq": (values - shift) ** 2})
            out[f"{stem}__{col}__count"] = grouped[col].count()
            out[f"{stem}__{col}__sum"] = centred.groupby(group_key)["__c"].sum()
            out[f"{stem}__{col}__sumsq"] = centred.groupby(group_key)["__csq"].sum()
            out[f"{stem}__{col}__shift"] = shift
            out[f"{stem}__{col}__min"] = grouped[col].min()
            out[f"{stem}__{col}__max"] = grouped[col].max()
        else:
            # nunique and mode are NOT semiring aggregates -- distinct-count cannot be
            # rolled up exactly without a sketch, and a mode of modes is not a mode.
            out[f"{stem}__{col}__nunique"] = grouped[col].nunique()
            # One vectorised pass instead of a Python callback per group, which
            # profiled at ~51% of total runtime. Counting (group, value) pairs and
            # taking the largest per group is the same answer.
            #
            # NaN is dropped by the count, so a group that is non-empty but entirely
            # null simply has no row here and reindexes to NaN -- which is what a
            # missing mode should be, and is the case that raised IndexError before.
            pair_counts = (
                df.groupby([group_key, series.name], dropna=True, observed=True)
                .size()
                .rename("__n")
                .reset_index()
            )
            if len(pair_counts):
                best = pair_counts.sort_values("__n", ascending=False).drop_duplicates(
                    subset=[group_key if isinstance(group_key, str) else pair_counts.columns[0]]
                )
                out[f"{stem}__{col}__mode"] = best.set_index(best.columns[0])[series.name]
            else:
                out[f"{stem}__{col}__mode"] = np.nan

    out[f"{stem}__count"] = out[f"{stem}__count"].fillna(0)
    return out


def _aggregate_by_key(child: Table, keys: pd.Series, cutoff: Optional[pd.Series]) -> pd.DataFrame:
    """Aggregate to one row per key. Requires keys to be unique (nested path)."""
    df, nested_stats = _resolve(
        child, pd.Series(cutoff.values, index=keys.values) if cutoff is not None else None
    )

    if cutoff is not None:
        if child.time_column is None:
            raise ValueError(f"table {child.name!r} needs time_column to honour a cutoff")
        per_row_cutoff = df[child.foreign_key].map(pd.Series(cutoff.values, index=keys.values))
        df = df[df[child.time_column] < per_row_cutoff]

    grouped = df.groupby(child.foreign_key)
    index = pd.Index(keys.values, name=child.foreign_key)
    return _stat_columns(child, df, nested_stats, grouped, index, child.foreign_key).reset_index(drop=True)


def _window_label(window) -> str:
    """A short, stable column-name fragment for a window."""
    try:
        days = pd.Timedelta(window).days
        return f"{days}d" if days else f"{int(pd.Timedelta(window).total_seconds())}s"
    except (TypeError, ValueError):
        return str(window).replace(" ", "")


def _aggregate_by_row(child: Table, anchor: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Aggregate to one row per *entity row*, not per key.

    The same entity usually appears many times with different prediction timestamps --
    in RelBench's rel-f1 a driver averages ~15 rows and reaches 59 -- so its features
    differ per row even though the key does not. Grouping by key would collapse those
    into one, and the cutoff filter would be ambiguous besides. So child rows are
    joined to entity *rows*, filtered against that row's own cutoff, and grouped by
    row position.

    Cost: the join is |child| x (rows sharing a key), which is the price of
    per-row-correct features.
    """
    df, nested_stats = _resolve(child, None)

    pairs = df.merge(anchor, left_on=child.foreign_key, right_on="__key", how="inner")
    if "__cutoff" in anchor.columns:
        if child.time_column is None:
            raise ValueError(f"table {child.name!r} needs time_column to honour a cutoff")
        pairs = pairs[pairs[child.time_column] < pairs["__cutoff"]]

    grouped = pairs.groupby("__row")
    stats = _stat_columns(child, pairs, nested_stats, grouped, grouped.size().index, "__row")
    stats = stats.reindex(range(n_rows))

    # Windowed blocks. Each restricts the same join to a recent slice, so a model can
    # see "lately" separately from "ever" rather than having to infer it from a mean
    # that mixes the two.
    for window in child.windows:
        if child.time_column is None or "__cutoff" not in pairs.columns:
            raise ValueError(
                f"table {child.name!r} has windows, which need time_column and a cutoff"
            )
        recent = pairs[pairs[child.time_column] >= pairs["__cutoff"] - window]
        label = _window_label(window)
        windowed = Table(
            df=child.df, foreign_key=child.foreign_key, name=f"{child.name}_{label}",
            time_column=child.time_column, columns=child.columns,
            primary_key=child.primary_key,
        )
        g = recent.groupby("__row")
        block = _stat_columns(windowed, recent, nested_stats, g, g.size().index, "__row")
        block = block.reindex(range(n_rows))
        own = f"{windowed.name}__count"
        if own in block.columns:
            block[own] = block[own].fillna(0)
        stats = pd.concat([stats, block], axis=1)

    # Reindexing introduces NaN for entity rows with no history. This table's own
    # count is genuinely zero there. Everything else stays unknown: a missing mean is
    # not 0, and a *nested* count belongs to a level that was never reached, so
    # claiming 0 would assert a fact about rows that do not exist.
    own_count = f"{child.name}__count"
    if own_count in stats.columns:
        stats[own_count] = stats[own_count].fillna(0)
    return stats.reset_index(drop=True)


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
    restored = {}
    for col in frame.columns:
        if not col.endswith("__sum"):
            continue
        base = col[: -len("__sum")]
        count_col = f"{base}__count"
        if count_col not in frame.columns:
            continue
        count = frame[count_col].replace(0, np.nan)
        # `sum` holds the CENTRED total; undo the pivot for the reported sum and mean,
        # but take the variance straight from the centred accumulators, which is where
        # the precision was preserved.
        shift = frame[f"{base}__shift"] if f"{base}__shift" in frame.columns else 0.0
        centred_mean = frame[col] / count
        derived[f"{base}__mean"] = centred_mean + shift
        restored[col] = frame[col] + count * shift
        sumsq_col = f"{base}__sumsq"
        if sumsq_col in frame.columns:
            var = (frame[sumsq_col] / count) - centred_mean**2
            derived[f"{base}__std"] = np.sqrt(var.clip(lower=0))

    if restored:
        frame = frame.assign(**restored)
    if derived:
        frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    carriers = [c for c in frame.columns if c.endswith("__sumsq") or c.endswith("__shift")]
    return frame.drop(columns=carriers)


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
    if cutoff_column and cutoff_column not in entity_df.columns:
        raise ValueError(f"cutoff_column {cutoff_column!r} not in entity_df")

    keys = entity_df[primary_key]
    duplicated = bool(keys.duplicated().any())
    if duplicated and any(child.children for child in children):
        raise ValueError(
            "entity keys repeat, which makes a grandchild's cutoff ambiguous; "
            "nested children currently require one row per entity key"
        )

    names = [child.name for child in children]
    clashes = {n for n in names if names.count(n) > 1}
    if clashes:
        # Every generated column is prefixed by the table name, so a collision yields
        # duplicate columns and pandas fails later with an opaque internal error.
        raise ValueError(f"child table names must be unique; repeated: {sorted(clashes)}")

    anchor = pd.DataFrame({"__key": keys.values, "__row": np.arange(len(entity_df))})
    if cutoff_column:
        anchor["__cutoff"] = entity_df[cutoff_column].values

    parts = [entity_df.drop(columns=[c for c in (primary_key, cutoff_column) if c]).reset_index(drop=True)]
    for child in children:
        parts.append(_aggregate_by_row(child, anchor, len(entity_df)))

    return _derive_features(pd.concat(parts, axis=1))


def asof_statistics(
    child: Table,
    keys: np.ndarray,
    cutoffs: np.ndarray,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Sufficient statistics per (key, cutoff) by scanning, not joining.

    :func:`flatten_relational` joins child rows to entity rows and groups, which costs
    ``|child| x (rows sharing a key)`` -- the product, not the sum. Sorting both sides
    once and walking them together is ``O(n log n)``.

    The window case is where this stops being a constant-factor argument. Because
    ``SUM_PRODUCT`` is invertible, the statistics over ``[t0, t1)`` are the prefix at
    ``t1`` minus the prefix at ``t0``: a window costs two lookups rather than a second
    pass over the data, and adding windows costs almost nothing. That is the
    ``invertible`` flag on the semiring doing real work rather than documenting a
    property.

    Covers the invertible statistics -- ``count``, ``sum``, ``sumsq``, and ``mean`` and
    ``std`` derived from them. ``min``/``max`` are deliberately absent: they are a
    semilattice, not a group, so a prefix difference cannot recover them and a range
    minimum needs a different structure. Use :func:`flatten_relational` when those or
    categorical statistics are wanted.

    Parameters
    ----------
    child : Table
        Must carry ``time_column``.

    keys, cutoffs : np.ndarray
        One entry per entity row. Keys may repeat, with different cutoffs.

    columns : Sequence[str], optional
        Numeric columns to aggregate. Defaults to every numeric column that is not the
        foreign key or the timestamp.

    Returns
    -------
    pd.DataFrame
        One row per entity row, in the given order, with all-history statistics and one
        block per entry in ``child.windows``.
    """
    if child.time_column is None:
        raise ValueError(f"table {child.name!r} needs time_column for an as-of scan")

    df = child.df
    if columns is None:
        excluded = {child.foreign_key, child.time_column, child.primary_key}
        columns = [
            c for c in df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]

    # One shared factorisation so child rows and entity rows agree on key identity.
    codes, _ = pd.factorize(np.concatenate([df[child.foreign_key].to_numpy(), keys]))
    child_key = codes[: len(df)]
    entity_key = codes[len(df):]

    order = np.lexsort((df[child.time_column].to_numpy(), child_key))
    sorted_key = child_key[order]
    sorted_time = df[child.time_column].to_numpy()[order]

    # Where each key's block begins, so a search can be confined to it.
    n_keys = codes.max() + 1 if len(codes) else 0
    starts = np.searchsorted(sorted_key, np.arange(n_keys + 1), side="left")

    prefixes = {"__count": np.arange(1, len(df) + 1, dtype=np.float64)}
    # Accumulate around a pivot (shifted-data variance). Invertibility is exact in the
    # reals but not in float: differencing two large cumulative sums cancels, and the
    # naive E[X^2] - E[X]^2 then loses every significant digit. Measured on data
    # offset by 1e6, the unshifted form returned std 4.35 against a true value near 1.
    # Subtracting the column mean first keeps the running sums near zero, which is
    # where double precision has digits to spare.
    shifts = {}
    for col in columns:
        values = df[col].to_numpy(dtype=np.float64)[order]
        shift = float(np.nanmean(values)) if len(values) else 0.0
        shifts[col] = shift
        centred = values - shift
        prefixes[f"{col}__sum"] = np.cumsum(centred)
        prefixes[f"{col}__sumsq"] = np.cumsum(centred**2)
    # Cumulative sums run across key boundaries, so every lookup is expressed as a
    # difference against the value at the key's own start -- which removes the offset.
    for name, arr in prefixes.items():
        prefixes[name] = np.concatenate([[0.0], arr])

    def upto(when: np.ndarray) -> np.ndarray:
        """Index (into the prefix arrays) of the last child row before ``when``."""
        lo = starts[entity_key]
        hi = starts[entity_key + 1]
        pos = np.array(
            [np.searchsorted(sorted_time[a:b], t, side="left") + a for a, b, t in zip(lo, hi, when)]
        )
        return pos

    out: dict[str, np.ndarray] = {}

    def block(label: str, hi_idx: np.ndarray, lo_idx: np.ndarray) -> None:
        base = starts[entity_key]
        n = prefixes["__count"][hi_idx] - prefixes["__count"][lo_idx]
        out[f"{label}__count"] = n
        for col in columns:
            d_sum = prefixes[f"{col}__sum"][hi_idx] - prefixes[f"{col}__sum"][lo_idx]
            d_sq = prefixes[f"{col}__sumsq"][hi_idx] - prefixes[f"{col}__sumsq"][lo_idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                safe = np.maximum(n, 1)
                centred_mean = d_sum / safe
                # Variance is shift-invariant, so it comes straight from the centred
                # accumulators; only the mean has to be shifted back.
                var = np.where(n > 0, d_sq / safe - centred_mean**2, np.nan)
                mean = np.where(n > 0, centred_mean + shifts[col], np.nan)
            out[f"{label}__{col}__sum"] = np.where(n > 0, d_sum + n * shifts[col], np.nan)
            out[f"{label}__{col}__mean"] = mean
            out[f"{label}__{col}__std"] = np.sqrt(np.clip(var, 0, None))
        del base

    hi = upto(cutoffs)
    block(child.name, hi, starts[entity_key])
    for window in child.windows:
        lo = upto(cutoffs - window)
        block(f"{child.name}_{_window_label(window)}", hi, lo)

    return pd.DataFrame(out)
