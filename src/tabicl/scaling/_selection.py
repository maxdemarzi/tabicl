"""Dropping features that cannot carry signal, before a model has to ignore them.

Adding relational history is not free. Across five RelBench tasks it helped on three
and *hurt* on two -- rel-event/user-ignore lost 0.102 AUC when seven aggregate columns
were added to six entity columns. Enriching the feature set did not fix it: adding
min/max/nunique moved that task by 0.0002. The columns are genuinely noise, not
under-specified.

The obvious cheap defence -- drop columns that are mostly missing or near-constant --
**does not work**, and the measurement is recorded here so it is not tried again as
though it were untested:

  task                    features   raw      pruned    delta
  rel-event/user-ignore   63 -> 3    0.5451   0.5032    -0.042
  rel-event/user-repeat   63 -> 3    0.5635   0.4684    -0.095
  rel-f1/driver-top3      198 -> 138 0.8926   0.8930    +0.000

At ``max_missing=0.5`` the sparse child table loses 60 of 63 columns and the task gets
worse. The premise was false: mostly-null aggregates over a sparse history are weak,
not worthless, and a model that can use them beats one denied them. On a dense schema
pruning is merely neutral.

So this is kept as a tool with deliberately conservative defaults -- it is useful for
stripping genuinely constant columns before an expensive fit -- and not as an answer
to relational features hurting. What distinguishes it from the approach that might
work is that it decides by heuristic; block-level selection decides by measuring
validation score with the model that will actually be used.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["prune_features"]


def prune_features(
    train: pd.DataFrame,
    others: Optional[List[pd.DataFrame]] = None,
    max_missing: float = 0.98,
    min_unique: int = 2,
    max_dominant: float = 0.99,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """Drop columns that are mostly missing, constant, or near-constant.

    Fitted on ``train`` only and applied to the rest, because a threshold chosen with
    validation data visible is a leak -- a quiet one, since it shows up as a better
    score rather than an error.

    Parameters
    ----------
    train : pd.DataFrame
        The matrix whose statistics decide what to keep.

    others : list of pd.DataFrame, optional
        Further matrices (validation, test) to apply the same decision to.

    max_missing : float, default=0.98
        Drop a column missing in more than this fraction of rows. The default is
        deliberately near-1: at 0.5 this removed 60 of 63 columns on rel-event and cost
        up to 0.095 AUC, because sparse aggregates still carry signal.

    min_unique : int, default=2
        Drop a column with fewer distinct non-null values than this. A constant column
        carries nothing by construction.

    max_dominant : float, default=0.99
        Drop a column where one value covers more than this fraction of rows. Catches
        the near-constant case a unique-count check misses -- 3999 zeros and one seven
        has two distinct values and no signal.

    Returns
    -------
    train_pruned, others_pruned, dropped
        The reduced matrices and the names removed, in input column order.
    """
    if not 0.0 <= max_missing <= 1.0:
        raise ValueError(f"max_missing must be in [0, 1], got {max_missing}")
    if not 0.0 < max_dominant <= 1.0:
        raise ValueError(f"max_dominant must be in (0, 1], got {max_dominant}")

    n_rows = len(train)
    dropped: List[str] = []
    for col in train.columns:
        series = train[col]
        if n_rows and series.isna().mean() > max_missing:
            dropped.append(col)
            continue
        non_null = series.dropna()
        if non_null.nunique() < min_unique:
            dropped.append(col)
            continue
        if len(non_null):
            share = non_null.value_counts(normalize=True).iloc[0]
            if share > max_dominant:
                dropped.append(col)

    keep = [c for c in train.columns if c not in set(dropped)]
    reduced = [df.reindex(columns=keep) for df in (others or [])]
    return train[keep], reduced, dropped
