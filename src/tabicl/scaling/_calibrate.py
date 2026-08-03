"""Choosing configuration per dataset, instead of hoping a default transfers.

Every lever measured in this project turned out to be task-dependent, several by
margins that dwarf any average effect:

    max_columns=2        +3.0 rel-event    0.0 rel-trial    -19.5 rel-f1
    categorical blocks   +1.25 rel-trial   0.0 rel-f1        -3.21 rel-event

Three separate attempts to find a good default failed the same way -- a result measured
on one task reversed on another. The conclusion is not that a better default exists; it
is that these settings should be *chosen* rather than assumed, on a validation split,
per dataset. That is what this module does.

Context size is the setting with the strongest cross-dataset support. Independent
evaluation across four temporal relational databases found quality curves shallow
enough that 10k-20k context rows land within epsilon of the best observed score, and a
measurement here agrees: rel-avito scores 63.30 at 5,000 context rows against 64.46 at
its full 116,598 -- 1.16 for a 23x smaller context, and the difference between needing a
46 GB GPU and not.

The selection rule matters as much as the sweep. Picking the argmax chases validation
noise and gives back the saving; these functions pick the **smallest** candidate within
``tolerance`` of the best, because the reason to shrink a context is cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Calibration", "calibrate_context_size", "sweep_configurations"]


@dataclass
class Calibration:
    """Outcome of a calibration sweep.

    Attributes
    ----------
    chosen : object
        The selected setting. Not the argmax -- the cheapest setting whose score is
        within ``tolerance`` of the best, where "cheapest" is the caller's ordering.

    best : object
        The argmax, kept so the cost of the tolerance is visible rather than hidden.

    curve : list of (setting, score)
        Every candidate evaluated, in the order given. Worth logging: a curve that is
        still climbing at the largest candidate means the sweep was too narrow, which a
        single chosen value would not reveal.

    tolerance : float
        The epsilon applied.
    """

    chosen: object
    best: object
    curve: List[Tuple[object, float]] = field(default_factory=list)
    tolerance: float = 0.0

    @property
    def best_score(self) -> float:
        return max(score for _, score in self.curve)

    @property
    def chosen_score(self) -> float:
        return next(score for setting, score in self.curve if setting == self.chosen)

    def __repr__(self) -> str:  # pragma: no cover - display only
        points = ", ".join(f"{setting}:{score:.4f}" for setting, score in self.curve)
        return (
            f"Calibration(chosen={self.chosen!r} ({self.chosen_score:.4f}), "
            f"best={self.best!r} ({self.best_score:.4f}), curve=[{points}])"
        )


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    random_state: Optional[int],
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take ``n`` rows, preserving the label distribution when asked.

    Stratification is not cosmetic at small budgets: rel-avito's positive rate is 0.905,
    so a uniform 1,000-row draw can leave a double-digit count of negatives and the
    score then measures the draw rather than the setting.
    """
    if n >= len(X):
        return X, y
    rng = np.random.default_rng(random_state)
    if not stratify:
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]

    keep: List[np.ndarray] = []
    for value in np.unique(y):
        pool = np.flatnonzero(y == value)
        take = max(1, int(round(n * len(pool) / len(y))))
        keep.append(pool[rng.permutation(len(pool))[:take]])
    idx = np.sort(np.concatenate(keep))
    return X[idx], y[idx]


def sweep_configurations(
    candidates: Sequence,
    score_fn: Callable[[object], float],
    tolerance: float = 0.0,
    cheaper_first: bool = True,
    feasible: Optional[Callable[[object], bool]] = None,
) -> Calibration:
    """Evaluate each candidate and select the cheapest one that is good enough.

    Parameters
    ----------
    candidates : Sequence
        Settings to try, ordered cheapest-first by default. The order defines what
        "cheapest" means; nothing here knows the cost of a setting.

    score_fn : callable
        ``score_fn(candidate) -> float``, higher is better. Must evaluate on data the
        final model will not be scored on.

    tolerance : float, default=0.0
        How much score the caller will trade for a cheaper setting. ``0.0`` selects the
        argmax; a small positive value is usually what is wanted, since the point of
        calibrating a cost knob is to spend less.

    cheaper_first : bool, default=True
        Whether ``candidates`` is already ordered cheapest-first. Set False when it is
        ordered the other way; the tolerance rule then still picks the cheap end.

    feasible : callable, optional
        ``feasible(candidate) -> bool``, consulted *before* the candidate is run. False
        scores it ``-inf`` without evaluating.

        This exists because catching exceptions is not a memory guard. A configuration
        that needs more memory than the machine has does not reliably raise
        ``MemoryError`` -- on a paging OS it thrashes instead, and a sweep that relies on
        try/except will take the machine down rather than skip the candidate. Predicting
        the cost and declining is the only thing that works. Measured: a rel-avito
        candidate reached 24.6 GB on a 5,000-row context, because the as-of scan builds
        prefix arrays over the full 5.3M-row child table regardless of context size.

    Returns
    -------
    Calibration
    """
    if not len(candidates):
        raise ValueError("candidates must be non-empty")

    def evaluate(candidate: object) -> float:
        if feasible is not None and not feasible(candidate):
            return float("-inf")
        return float(score_fn(candidate))

    curve = [(candidate, evaluate(candidate)) for candidate in candidates]
    best_setting, best_score = max(curve, key=lambda pair: pair[1])
    if best_score == float("-inf"):
        raise RuntimeError(
            "no candidate was runnable; widen the budget or supply cheaper candidates"
        )

    ordered = curve if cheaper_first else list(reversed(curve))
    chosen = next(
        (setting for setting, score in ordered if score >= best_score - tolerance),
        best_setting,
    )
    return Calibration(chosen=chosen, best=best_setting, curve=curve, tolerance=tolerance)


def calibrate_context_size(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fit_score: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    candidates: Sequence[Optional[int]] = (1000, 5000, 10000, 20000, None),
    tolerance: float = 0.005,
    random_state: Optional[int] = 0,
    stratify: bool = True,
) -> Calibration:
    """Pick how many training rows to keep in context.

    TabICL attends to the whole training set at once, so compute grows quadratically in
    context rows and peak memory linearly. Shrinking the context is the cheapest lever
    on both, and quality curves on relational data are shallow enough that it is usually
    close to free.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        The candidate context pool.

    X_val, y_val : np.ndarray
        Held out for selection. Never the test split -- a size chosen on test is a size
        fitted to test.

    fit_score : callable
        ``fit_score(X_ctx, y_ctx, X_val, y_val) -> float``, higher is better. Supplying
        this rather than an estimator keeps the module free of any assumption about the
        task, the metric, or the model.

    candidates : Sequence[Optional[int]], default=(1000, 5000, 10000, 20000, None)
        Context sizes to try, smallest first. ``None`` means the full training set and
        belongs last. Sizes at or above the pool size are evaluated once and collapse
        to the full set.

    tolerance : float, default=0.005
        Score the caller will trade for a smaller context. The default matches the
        epsilon used in the published sampling study for ROC-AUC.

    random_state : int, optional
        Seed for subsampling.

    stratify : bool, default=True
        Preserve the label distribution when subsampling.

    Returns
    -------
    Calibration
        ``chosen`` is the smallest context within ``tolerance`` of the best.

    Examples
    --------
    >>> def fit_score(Xc, yc, Xv, yv):        # doctest: +SKIP
    ...     model = TabICLClassifier().fit(Xc, yc)
    ...     return roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
    >>> calibrate_context_size(X_tr, y_tr, X_va, y_va, fit_score)   # doctest: +SKIP
    """
    seen: set = set()
    ordered: List[Optional[int]] = []
    for candidate in candidates:
        size = len(X_train) if candidate is None else min(int(candidate), len(X_train))
        if size in seen:
            continue
        seen.add(size)
        ordered.append(candidate)

    def score_fn(candidate: Optional[int]) -> float:
        n = len(X_train) if candidate is None else int(candidate)
        X_ctx, y_ctx = _subsample(X_train, y_train, n, random_state, stratify)
        return fit_score(X_ctx, y_ctx, X_val, y_val)

    return sweep_configurations(ordered, score_fn, tolerance=tolerance, cheaper_first=True)
