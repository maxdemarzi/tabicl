"""Test-time compute scaling for TabICL.

TabPFN-3's "Thinking mode" (arXiv 2605.13986, Section 2.6) is described only as
applying "additional inference-time computation"; the mechanism is undisclosed and
ships API-only. There is nothing to port, so this is an original design in the same
spirit: spend more compute at predict time, with the backbone frozen.

Two knobs, both cheap relative to a forward pass:

1. ``n_permutations`` -- average predictions over several TabICL estimator
   permutations, reducing variance from feature/sample ordering.
2. ``refit_head`` -- fit a light classifier on the frozen backbone's own predictive
   distribution and blend it in.

The blend weight is chosen on a held-out split, and 0.0 is always a candidate, so
thinking cannot score worse than the base model on that split. That guard matters:
extra compute that sometimes hurts is not a usable feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

__all__ = ["ThinkingResult", "think_predict_proba"]

_WEIGHT_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)


@dataclass
class ThinkingResult:
    """Outcome of a thinking-mode prediction."""

    proba: np.ndarray
    blend_weight: float
    n_permutations: int
    val_score_base: float
    val_score_thought: float

    @property
    def improved(self) -> bool:
        return self.val_score_thought > self.val_score_base


def _log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))


def _mean_proba(estimator, X_train, y_train, X_eval, n_permutations: int, rng) -> np.ndarray:
    """Average predicted probabilities over several random states."""
    total = None
    for i in range(n_permutations):
        est = clone(estimator)
        if "random_state" in est.get_params():
            est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
        est.fit(X_train, y_train)
        p = est.predict_proba(X_eval)
        total = p if total is None else total + p
    return total / n_permutations


def think_predict_proba(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_permutations: int = 3,
    refit_head: bool = True,
    val_size: float = 0.25,
    random_state: Optional[int] = 0,
) -> ThinkingResult:
    """Predict with extra test-time compute, guarded by a validation split.

    Parameters
    ----------
    estimator : TabICLClassifier
        Unfitted (or refittable) classifier. Cloned internally; never mutated.

    X_train, y_train : array-like
        Training context.

    X_test : array-like
        Rows to predict.

    n_permutations : int, default=3
        Estimator refits to average over. 1 disables averaging.

    refit_head : bool, default=True
        Whether to fit a logistic head on the frozen backbone's probabilities and
        blend it in.

    val_size : float, default=0.25
        Held-out fraction used to pick the blend weight.

    random_state : Optional[int], default=0
        Seed for the split and the permutation seeds.

    Returns
    -------
    ThinkingResult
        Final probabilities plus the diagnostics behind the choice.
    """
    rng = np.random.default_rng(random_state)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    classes, y_enc = np.unique(y_train, return_inverse=True)
    n_classes = len(classes)

    stratify = y_enc if np.bincount(y_enc).min() >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_enc, test_size=val_size, random_state=random_state, stratify=stratify
    )

    # Base: a single forward pass, the thing we must not regress against.
    base = clone(estimator)
    base.fit(X_fit, y_fit)
    val_base = base.predict_proba(X_val)
    score_base = _log_loss(y_val, val_base)

    # Knob 1: permutation averaging.
    val_thought = (
        _mean_proba(estimator, X_fit, y_fit, X_val, n_permutations, rng) if n_permutations > 1 else val_base
    )

    # Knob 2: a light head over the backbone's own distribution.
    best_w = 0.0
    if refit_head and len(X_fit) > 4 * n_classes:
        inner_fit, inner_cal, y_inner_fit, y_inner_cal = train_test_split(
            X_fit,
            y_fit,
            test_size=0.3,
            random_state=random_state,
            stratify=y_fit if np.bincount(y_fit).min() >= 2 else None,
        )
        inner = clone(estimator)
        inner.fit(inner_fit, y_inner_fit)
        head = LogisticRegression(max_iter=1000)
        try:
            head.fit(inner.predict_proba(inner_cal), y_inner_cal)
            head_val = head.predict_proba(val_thought)
            # 0.0 is in the grid, so the search can always fall back to no blending
            # -- that is the guard against thinking making things worse.
            scored = [(_log_loss(y_val, (1 - w) * val_thought + w * head_val), w) for w in _WEIGHT_GRID]
            best_w = min(scored)[1]
        except ValueError:
            best_w = 0.0

    score_thought = _log_loss(y_val, val_thought)

    # Refit on the full training set for the actual prediction.
    final = (
        _mean_proba(estimator, X_train, y_enc, X_test, n_permutations, rng)
        if n_permutations > 1
        else clone(estimator).fit(X_train, y_enc).predict_proba(X_test)
    )

    if best_w > 0:
        calib = clone(estimator)
        calib.fit(X_fit, y_fit)
        head = LogisticRegression(max_iter=1000)
        try:
            head.fit(calib.predict_proba(X_val), y_val)
            final = (1 - best_w) * final + best_w * head.predict_proba(final)
        except ValueError:
            pass

    return ThinkingResult(
        proba=final,
        blend_weight=best_w,
        n_permutations=n_permutations,
        val_score_base=score_base,
        val_score_thought=min(score_base, score_thought),
    )
