"""Regression tests: a column with no observed values must not crash prediction.

sklearn's SimpleImputer drops all-missing columns during fit by default. The estimators
build a `feature_mask` of all-NaN columns in the ORIGINAL feature space, specifically so
those columns can be masked out -- but the imputer had already removed the column, the two
disagreed by one, and predict raised

    IndexError: boolean index did not match indexed array along axis 0

Found on OpenML `sick` (in OpenML-CC18), whose TBG column is entirely missing. Pre-existing:
it reproduces on the code before any change on this branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabicl import TabICLClassifier, TabICLRegressor


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "empty": np.full(n, np.nan),          # the TBG case
        "c": rng.choice(list("xyz"), n),
    })
    return X, rng


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_classifier_predicts_with_an_all_nan_column(position):
    X, rng = _frame()
    cols = list(X.columns)
    cols.remove("empty")
    idx = {"first": 0, "middle": len(cols) // 2, "last": len(cols)}[position]
    cols.insert(idx, "empty")
    X = X[cols]
    y = (X["a"] > 0).astype(int).to_numpy()

    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    clf.fit(X.iloc[:60], y[:60])
    proba = clf.predict_proba(X.iloc[60:])
    assert proba.shape == (20, 2)
    assert np.isfinite(proba).all()


def test_regressor_predicts_with_an_all_nan_column():
    X, rng = _frame()
    y = X["a"].to_numpy() * 2.0 + rng.standard_normal(len(X)) * 0.1
    reg = TabICLRegressor(n_estimators=1, device="cpu", random_state=0)
    reg.fit(X.iloc[:60], y[:60])
    pred = reg.predict(X.iloc[60:])
    assert pred.shape == (20,) and np.isfinite(pred).all()


def test_numpy_input_with_an_all_nan_column():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    X[:, 2] = np.nan
    y = (X[:, 0] > 0).astype(int)
    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    clf.fit(X[:60], y[:60])
    assert clf.predict(X[60:]).shape == (20,)


def test_several_all_nan_columns():
    X, rng = _frame()
    X["empty2"] = np.nan
    X["empty3"] = np.nan
    y = (X["a"] > 0).astype(int).to_numpy()
    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    clf.fit(X.iloc[:60], y[:60])
    assert clf.predict(X.iloc[60:]).shape == (20,)


def test_column_that_is_empty_only_in_the_training_split():
    """Observed at test time, never at fit time -- the same code path from the other side."""

    X, rng = _frame()
    X["late"] = np.nan
    X.loc[X.index[60:], "late"] = rng.standard_normal(20)
    y = (X["a"] > 0).astype(int).to_numpy()
    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    clf.fit(X.iloc[:60], y[:60])
    assert np.isfinite(clf.predict_proba(X.iloc[60:])).all()
