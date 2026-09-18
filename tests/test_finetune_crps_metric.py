"""Tests for the ``crps`` eval_metric on FinetunedTabICLRegressor (TP-14).

The class trains against pinball loss over the raw quantile head
(``_compute_batch_loss``), so offering only mse/mae/r2 for early stopping means selection
and objective disagree. This adds a distributional option so they can agree.

Read the ledger before assuming it helps: on the one dataset where it was A/B tested it was
neutral-to-slightly-worse. It is kept because selecting on the metric you trained against is
principled and the option costs nothing, not because it was shown to win.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tabicl._finetune.regressor import _crps_from_quantiles


def test_crps_matches_closed_form_for_a_standard_normal():
    levels = np.linspace(1e-4, 1 - 1e-4, 4000)
    quantiles = stats.norm.ppf(levels).reshape(1, -1)
    expected = 2 * stats.norm.pdf(0.0) - 1.0 / np.sqrt(np.pi)
    assert _crps_from_quantiles(np.array([0.0]), quantiles, levels) == pytest.approx(expected, rel=2e-3)


def test_crps_of_a_point_mass_converges_to_absolute_error():
    levels = np.linspace(1e-4, 1 - 1e-4, 4000)
    quantiles = np.full((1, levels.size), 5.0)
    assert _crps_from_quantiles(np.array([2.0]), quantiles, levels) == pytest.approx(3.0, abs=1e-2)


def test_crps_rewards_the_sharper_correct_distribution():
    levels = np.linspace(0.01, 0.99, 99)
    sharp = stats.norm.ppf(levels, scale=0.5).reshape(1, -1)
    broad = stats.norm.ppf(levels, scale=3.0).reshape(1, -1)
    y = np.array([0.0])
    assert _crps_from_quantiles(y, sharp, levels) < _crps_from_quantiles(y, broad, levels)


def test_crps_averages_over_samples():
    levels = np.linspace(0.01, 0.99, 99)
    q = np.tile(stats.norm.ppf(levels), (5, 1))
    one = _crps_from_quantiles(np.zeros(1), q[:1], levels)
    five = _crps_from_quantiles(np.zeros(5), q, levels)
    assert five == pytest.approx(one, rel=1e-9)


def test_regressor_accepts_crps_as_eval_metric():
    from tabicl import FinetunedTabICLRegressor

    reg = FinetunedTabICLRegressor(eval_metric="crps")
    assert reg.eval_metric == "crps"
    assert reg._metric_name == "crps"


def test_regressor_still_rejects_an_unknown_eval_metric():
    """The new branch must not have turned the guard into a silent pass-through."""

    from tabicl import FinetunedTabICLRegressor

    reg = FinetunedTabICLRegressor(eval_metric="nonsense")

    class _Inner:
        def fit(self, X, y): return self
        def predict(self, X, **kw): return np.zeros(len(X))

    with pytest.raises(ValueError, match="Unsupported eval_metric"):
        reg._run_validation(_Inner(), np.zeros((4, 2)), np.zeros(4),
                                        np.zeros((4, 2)), np.zeros(4))


def test_crps_selection_reports_crps_in_secondary_metrics():
    """The selected metric must be visible in the log, not just used internally."""

    from tabicl import FinetunedTabICLRegressor

    reg = FinetunedTabICLRegressor(eval_metric="crps")
    levels_n = 99

    class _Inner:
        def fit(self, X, y): return self
        def predict(self, X, output_type=None, alphas=None):
            if output_type == "quantiles":
                return np.tile(np.linspace(-2, 2, len(alphas)), (len(X), 1))
            return np.zeros(len(X))

    out = reg._run_validation(_Inner(), np.zeros((6, 2)), np.zeros(6),
                                          np.zeros((6, 2)), np.zeros(6))
    assert "crps" in out.secondary
    assert out.primary == pytest.approx(-out.secondary["crps"])
    assert {"mse", "mae", "r2"} <= set(out.secondary)
    assert levels_n == 99


def test_crps_selection_survives_a_failing_quantile_prediction():
    """A model that cannot produce quantiles must not abort the whole finetune run."""

    from tabicl import FinetunedTabICLRegressor

    reg = FinetunedTabICLRegressor(eval_metric="crps")

    class _Inner:
        def fit(self, X, y): return self
        def predict(self, X, output_type=None, alphas=None):
            if output_type == "quantiles":
                raise RuntimeError("no quantile head")
            return np.zeros(len(X))

    out = reg._run_validation(_Inner(), np.zeros((4, 2)), np.zeros(4),
                                          np.zeros((4, 2)), np.zeros(4))
    assert np.isnan(out.primary)
    assert "mse" in out.secondary
