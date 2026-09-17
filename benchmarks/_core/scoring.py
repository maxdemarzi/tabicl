"""Proper scoring rules for predictive distributions.

For TP-14. ScoringBench evaluates the *full* predicted distribution of a regressor rather
than its point error, and its headline metric is the continuous ranked probability score.
This matters here because the report has TabICLv2 at CRPS mean rank 12.05 against TabPFN-3
at 7.64 -- an outlier relative to every other board, on a capability this project has
natively (``_model/quantile_dist.py``).

CRPS from a set of predictive quantiles uses the pinball identity: for quantile levels
:math:`\\tau_1 < ... < \\tau_K` with predictions :math:`q_k`,

.. math::
    \\mathrm{CRPS}(F, y) \\approx 2 \\int_0^1 \\mathrm{PB}_\\tau(q_\\tau, y)\\, d\\tau

where :math:`\\mathrm{PB}` is the pinball loss. The factor of 2 and the trapezoidal
integration over :math:`\\tau` are the parts most often dropped, and dropping either yields a
number that ranks the same but is not comparable to a published one -- which is the whole
point of running this benchmark.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pinball_loss", "crps_from_quantiles", "interval_score", "coverage"]


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray:
    """Per-sample pinball (quantile) loss at level ``tau``."""

    diff = y_true - q_pred
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def crps_from_quantiles(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
    reduce: bool = True,
):
    """CRPS approximated from predictive quantiles.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)

    quantiles : ndarray of shape (n_samples, n_levels)
        Predicted quantiles, one column per level.

    levels : ndarray of shape (n_levels,)
        Quantile levels in (0, 1), strictly increasing.

    reduce : bool, default=True
        Return the mean over samples rather than the per-sample vector.

    Returns
    -------
    float or ndarray
        CRPS, lower is better.

    Notes
    -----
    The integral is evaluated by the trapezoidal rule over ``levels``. With a coarse or
    narrow grid this *underestimates* CRPS, because the tails beyond the outermost levels
    contribute nothing -- so two models must be scored on the same grid for the comparison
    to mean anything.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)

    if quantiles.ndim != 2 or quantiles.shape[1] != levels.size:
        raise ValueError(
            f"quantiles must be (n_samples, n_levels); got {quantiles.shape} for {levels.size} levels"
        )
    if quantiles.shape[0] != y_true.size:
        raise ValueError(f"y_true has {y_true.size} samples but quantiles has {quantiles.shape[0]}")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be strictly increasing")

    losses = np.stack(
        [pinball_loss(y_true, quantiles[:, k], float(levels[k])) for k in range(levels.size)],
        axis=1,
    )
    per_sample = 2.0 * np.trapezoid(losses, x=levels, axis=1)
    return float(per_sample.mean()) if reduce else per_sample


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float, reduce: bool = True
):
    """Winkler interval score for a central ``1 - alpha`` prediction interval.

    Width plus a penalty for each observation falling outside. Lower is better.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)

    width = upper - lower
    below = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    above = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    per_sample = width + below + above
    return float(per_sample.mean()) if reduce else per_sample


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of an interval.

    Reported alongside :func:`interval_score` because they fail in opposite directions: a
    degenerate interval scores badly but covers nothing, and an enormous one covers
    everything while being useless. Neither number is interpretable alone.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    inside = (y_true >= np.asarray(lower).reshape(-1)) & (y_true <= np.asarray(upper).reshape(-1))
    return float(inside.mean())
