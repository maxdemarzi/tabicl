"""Refusals that make a leak impossible to report, rather than possible to notice.

This project's recurring failure is not a wrong number, it is a *plausible* wrong number.
Every leak found so far was found by someone looking at a figure and thinking it looked
odd -- a per-split factorize, an untimed link table, a cutoff that never reached the
grandchildren, and a target column passed straight through as a feature, which scored
AUC 100.00 in both arms of a paired comparison and would have read as "no effect".

That last one is the argument for this module. A guard that fires on the *symptom* --
a feature that separates the target perfectly -- does not need to know which mistake
produced it, and catches the target under a renamed column, a duplicated column, or an
aggregate that happens to reconstruct it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ["assert_no_perfect_feature"]

# Below 1.0 because a genuinely strong feature is possible and this must not fire on one;
# far enough above any real single-column AUC seen on these tasks (the best measured is
# around 82) that anything reaching it is a mistake rather than a result.
PERFECT_AUC = 99.5


def assert_no_perfect_feature(
    X: np.ndarray,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    threshold: float = PERFECT_AUC,
    context: str = "",
) -> None:
    """Raise if any single column predicts ``y`` essentially perfectly.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, one row per label.

    y : np.ndarray
        Binary labels.

    columns : Sequence[str], optional
        Column names, used only to name the offender in the message.

    threshold : float, default=99.5
        AUC x100 at or above which a column is treated as a leak.

    context : str, default=""
        Free text describing what was being built, prepended to the message.

    Raises
    ------
    SystemExit
        Naming the column, so the fix is to the featuriser and not to the threshold.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return
    offenders = []
    for j in range(X.shape[1]):
        values = X[:, j]
        if not np.isfinite(values).all() or len(np.unique(values)) < 2:
            continue
        auc = roc_auc_score(y, values) * 100
        if max(auc, 100 - auc) >= threshold:
            name = columns[j] if columns is not None and j < len(columns) else f"column {j}"
            offenders.append((name, max(auc, 100 - auc)))
    if offenders:
        listed = ", ".join(f"{name} ({auc:.2f})" for name, auc in offenders[:5])
        raise SystemExit(
            f"{context + ': ' if context else ''}{len(offenders)} feature(s) predict the "
            f"target essentially alone -- {listed}. That is a leak, not a result: the "
            f"target column, a copy of it, or an aggregate that reconstructs it has "
            f"reached the feature frame. Fix the featuriser; do not raise the threshold."
        )
