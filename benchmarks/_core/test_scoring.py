"""Tests for the proper scoring rules.

CRPS is pinned against cases with a known closed form. This matters more than usual: TP-14
exists because our published CRPS rank looks anomalous, and a scoring bug in the harness
would be indistinguishable from the model weakness we are trying to rule out.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from benchmarks._core.scoring import coverage, crps_from_quantiles, interval_score, pinball_loss


def test_pinball_is_asymmetric_in_the_right_direction():
    # tau = 0.9 penalises under-prediction more than over-prediction.
    under = pinball_loss(np.array([1.0]), np.array([0.0]), 0.9)
    over = pinball_loss(np.array([0.0]), np.array([1.0]), 0.9)
    assert under > over


def test_pinball_at_the_median_is_half_absolute_error():
    y, q = np.array([3.0]), np.array([1.0])
    assert pinball_loss(y, q, 0.5)[0] == pytest.approx(0.5 * abs(3.0 - 1.0))


def test_crps_of_a_point_mass_converges_to_absolute_error():
    """A distribution concentrated at c scores |c - y| in the limit of a full level grid.

    On a truncated grid it comes in *under* that, by exactly the mass the trapezoid misses
    outside [levels[0], levels[-1]] -- which is the bias documented on
    :func:`crps_from_quantiles` and the reason two models must share a grid. Asserting the
    convergence rather than a single tolerance keeps that property visible instead of hiding
    it behind a loose `approx`.
    """

    y = np.array([2.0])
    exact = 3.0

    errors = []
    for eps in (1e-2, 1e-3, 1e-4):
        levels = np.linspace(eps, 1 - eps, 2001)
        quantiles = np.full((1, levels.size), 5.0)
        value = crps_from_quantiles(y, quantiles, levels)
        assert value < exact  # truncation can only lose mass
        errors.append(exact - value)

    # Each tenfold narrowing of the truncated tails cuts the error by about ten.
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] == pytest.approx(0.0, abs=1e-3)


def test_crps_truncation_bias_is_the_missing_tail_mass():
    """Pin the size of the bias, so a change in the integration is not silent.

    For a point mass the pinball loss is linear in tau, so the trapezoid over
    [a, 1-a] is exactly (1 - 2a) of the full integral.
    """

    a = 0.01
    levels = np.linspace(a, 1 - a, 5001)
    quantiles = np.full((1, levels.size), 5.0)
    value = crps_from_quantiles(np.array([2.0]), quantiles, levels)
    assert value == pytest.approx(3.0 * (1 - 2 * a), rel=1e-4)


def test_crps_matches_the_closed_form_for_a_standard_normal():
    """CRPS(N(0,1), 0) = 2/sqrt(pi) - 1/sqrt(pi) ... = 1/sqrt(pi) * (2/sqrt(2) ... )

    Using the standard result CRPS(N(mu, sigma), y) with z = (y-mu)/sigma:
        sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    At y = mu = 0, sigma = 1 this is 2*phi(0) - 1/sqrt(pi).
    """

    levels = np.linspace(1e-4, 1 - 1e-4, 4000)
    quantiles = stats.norm.ppf(levels).reshape(1, -1)
    expected = 2 * stats.norm.pdf(0.0) - 1.0 / np.sqrt(np.pi)
    assert crps_from_quantiles(np.array([0.0]), quantiles, levels) == pytest.approx(expected, rel=2e-3)


def test_crps_rewards_a_sharper_correct_distribution():
    levels = np.linspace(0.01, 0.99, 99)
    y = np.array([0.0])
    sharp = stats.norm.ppf(levels, scale=0.5).reshape(1, -1)
    broad = stats.norm.ppf(levels, scale=3.0).reshape(1, -1)
    assert crps_from_quantiles(y, sharp, levels) < crps_from_quantiles(y, broad, levels)


def test_crps_punishes_a_confidently_wrong_distribution():
    levels = np.linspace(0.01, 0.99, 99)
    y = np.array([0.0])
    confident_wrong = stats.norm.ppf(levels, loc=5.0, scale=0.2).reshape(1, -1)
    humble_wrong = stats.norm.ppf(levels, loc=5.0, scale=3.0).reshape(1, -1)
    assert crps_from_quantiles(y, confident_wrong, levels) > crps_from_quantiles(y, humble_wrong, levels)


def test_crps_per_sample_shape():
    levels = np.linspace(0.1, 0.9, 9)
    quantiles = np.tile(stats.norm.ppf(levels), (7, 1))
    out = crps_from_quantiles(np.zeros(7), quantiles, levels, reduce=False)
    assert out.shape == (7,)


def test_crps_rejects_unsorted_levels():
    with pytest.raises(ValueError, match="strictly increasing"):
        crps_from_quantiles(np.array([0.0]), np.zeros((1, 3)), np.array([0.5, 0.1, 0.9]))


def test_crps_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="n_samples, n_levels"):
        crps_from_quantiles(np.array([0.0]), np.zeros((1, 4)), np.array([0.1, 0.5, 0.9]))


def test_interval_score_penalises_misses():
    hit = interval_score(np.array([0.0]), np.array([-1.0]), np.array([1.0]), alpha=0.1)
    miss = interval_score(np.array([5.0]), np.array([-1.0]), np.array([1.0]), alpha=0.1)
    assert miss > hit
    assert hit == pytest.approx(2.0)  # pure width when covered


def test_interval_score_prefers_the_narrower_of_two_covering_intervals():
    narrow = interval_score(np.array([0.0]), np.array([-1.0]), np.array([1.0]), alpha=0.1)
    wide = interval_score(np.array([0.0]), np.array([-10.0]), np.array([10.0]), alpha=0.1)
    assert narrow < wide


def test_coverage_counts_inclusively():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert coverage(y, np.zeros(4), np.full(4, 2.0)) == pytest.approx(0.75)
