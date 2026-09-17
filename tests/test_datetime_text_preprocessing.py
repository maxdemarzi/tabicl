"""Tests for TP-12: native datetime and text preprocessing.

Datetime columns used to be dropped outright -- neither the numeric nor the string selector
in ``TransformToNumerical`` matches ``datetime64`` -- so the tests here mostly assert that
information now survives the pipeline that previously left it behind.

Text encoding is opt-in (``text_encoding="tfidf"``); the default stays ordinal so that
existing behaviour is unchanged until STRABLE says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabicl._sklearn.preprocessing import (
    DatetimeEncoder,
    TextEncoder,
    TransformToNumerical,
    classify_string_columns,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "num": rng.standard_normal(n),
            "cat": rng.choice(list("abc"), n),
            "when": pd.date_range("2021-03-04", periods=n, freq="9h"),
            "note": [f"customer reported a {w} fault on unit {i}" for i, w in
                     enumerate(rng.choice(["hydraulic", "electrical", "thermal"], n))],
        }
    )


# --------------------------------------------------------------------------- #
# DatetimeEncoder
# --------------------------------------------------------------------------- #


def test_datetime_encoder_emits_features(frame):
    enc = DatetimeEncoder().fit(frame)
    out = enc.transform(frame)
    assert out.shape[0] == len(frame)
    assert out.shape[1] == len(enc.feature_names_) > 0


def test_datetime_encoder_emits_monotonic_epoch(frame):
    """The trend term. This is what a temporal split needs and an ordinal month cannot give."""

    enc = DatetimeEncoder().fit(frame)
    out = enc.transform(frame)
    epoch = out[:, enc.feature_names_.index("when__epoch")]
    assert np.all(np.diff(epoch) > 0)


def test_datetime_encoder_cyclical_pairs_are_on_the_unit_circle(frame):
    enc = DatetimeEncoder().fit(frame)
    out = enc.transform(frame)
    names = enc.feature_names_
    for part in ("month", "dayofweek", "hour"):
        if f"when__{part}_sin" not in names:
            continue
        sin = out[:, names.index(f"when__{part}_sin")]
        cos = out[:, names.index(f"when__{part}_cos")]
        np.testing.assert_allclose(sin**2 + cos**2, 1.0, atol=1e-9)


def test_datetime_encoder_december_and_january_are_adjacent():
    """The reason cyclical encoding exists: as integers they are 11 apart."""

    df = pd.DataFrame({"d": pd.to_datetime(["2021-12-15", "2022-01-15", "2021-06-15"])})
    enc = DatetimeEncoder(drop_constant=False).fit(df)
    out = enc.transform(df)
    names = enc.feature_names_
    circle = np.stack([out[:, names.index("d__month_sin")], out[:, names.index("d__month_cos")]], 1)

    dec_jan = np.linalg.norm(circle[0] - circle[1])
    dec_jun = np.linalg.norm(circle[0] - circle[2])
    assert dec_jan < dec_jun


def test_datetime_encoder_drops_constant_features():
    """A date-only column must not contribute four dead hour features."""

    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"])})
    enc = DatetimeEncoder(drop_constant=True).fit(df)
    assert not any("hour" in name for name in enc.feature_names_)

    keep_all = DatetimeEncoder(drop_constant=False).fit(df)
    assert any("hour" in name for name in keep_all.feature_names_)


def test_datetime_encoder_missing_becomes_nan_not_a_sentinel():
    """NaT casts to a huge negative int64; left alone it would read as a real date."""

    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", None, "2021-03-01"])})
    enc = DatetimeEncoder(drop_constant=False).fit(df)
    out = enc.transform(df)
    epoch = out[:, enc.feature_names_.index("d__epoch")]
    assert np.isnan(epoch[1])
    assert np.isfinite(epoch[0]) and np.isfinite(epoch[2])


def test_datetime_encoder_with_no_datetime_columns_is_empty():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    enc = DatetimeEncoder().fit(df)
    assert enc.transform(df).shape == (2, 0)


def test_datetime_encoder_transform_matches_fit_width(frame):
    enc = DatetimeEncoder().fit(frame)
    assert enc.transform(frame.iloc[:10]).shape[1] == enc.transform(frame).shape[1]


# --------------------------------------------------------------------------- #
# TextEncoder
# --------------------------------------------------------------------------- #


def test_text_encoder_shape_and_determinism(frame):
    enc = TextEncoder(n_components=8, random_state=0).fit(frame[["note"]])
    a = enc.transform(frame[["note"]])
    b = enc.transform(frame[["note"]])
    assert a.shape == (len(frame), 8)
    np.testing.assert_allclose(a, b)


def test_text_encoder_puts_similar_strings_closer_than_dissimilar():
    """The property ordinal encoding destroys: shared substrings carry no signal there."""

    df = pd.DataFrame({"t": ["Acme Corporation", "Acme Corp", "Zeta Industries Limited"]})
    enc = TextEncoder(n_components=2, random_state=0).fit(df)
    out = enc.transform(df)
    same = np.linalg.norm(out[0] - out[1])
    other = np.linalg.norm(out[0] - out[2])
    assert same < other


def test_text_encoder_handles_tiny_vocabulary():
    """SVD cannot ask for more components than the TF-IDF matrix has columns."""

    df = pd.DataFrame({"t": ["ab", "ab", "ba"]})
    enc = TextEncoder(n_components=30, random_state=0).fit(df)
    out = enc.transform(df)
    assert out.shape[0] == 3 and 1 <= out.shape[1] <= 30


def test_text_encoder_handles_unseen_values_at_transform_time():
    train = pd.DataFrame({"t": ["alpha widget", "beta widget"]})
    enc = TextEncoder(n_components=2, random_state=0).fit(train)
    out = enc.transform(pd.DataFrame({"t": ["gamma sprocket"]}))
    assert out.shape == (1, 2) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# classify_string_columns
# --------------------------------------------------------------------------- #


def test_classify_splits_on_cardinality_and_length():
    df = pd.DataFrame(
        {
            "low_card": ["a", "b", "c"] * 40,
            "long_text": [f"a reasonably long sentence number {i}" for i in range(120)],
        }
    )
    cat, text = classify_string_columns(df, ["low_card", "long_text"])
    assert cat == ["low_card"]
    assert text == ["long_text"]


def test_classify_treats_high_cardinality_short_ids_as_categorical():
    """An identifier column is not free text; n-grams over it produce noise features."""

    df = pd.DataFrame({"id": [f"{i:05d}" for i in range(200)]})
    cat, text = classify_string_columns(df, ["id"])
    assert cat == ["id"] and text == []


# --------------------------------------------------------------------------- #
# TransformToNumerical wiring
# --------------------------------------------------------------------------- #


def test_datetime_survives_the_pipeline_by_default(frame):
    """Previously this column was dropped without comment."""

    with_dt = TransformToNumerical().fit(frame).transform(frame)
    without = TransformToNumerical(datetime_features=False).fit(frame).transform(frame)
    assert with_dt.shape[1] > without.shape[1]


def test_default_text_encoding_is_unchanged(frame):
    """Default stays ordinal: TP-12 must not silently change existing behaviour."""

    tfm = TransformToNumerical()
    with pytest.warns(UserWarning, match="cardinality above"):
        tfm.fit(frame)
    assert [name for name, _, _ in tfm.tfm_.transformers_] == ["categorical", "continuous", "datetime"]


def test_tfidf_encoding_adds_a_text_branch(frame):
    tfm = TransformToNumerical(text_encoding="tfidf", text_n_components=5, random_state=0).fit(frame)
    names = [name for name, _, _ in tfm.tfm_.transformers_]
    assert "text" in names
    assert tfm.transform(frame).shape[1] == TransformToNumerical().fit(frame).transform(frame).shape[1] - 1 + 5


def test_pipeline_output_is_finite_and_numeric(frame):
    out = TransformToNumerical(text_encoding="tfidf", random_state=0).fit(frame).transform(frame)
    assert out.dtype.kind == "f"
    assert np.isfinite(out).all()


def test_numpy_input_path_is_untouched():
    """The non-DataFrame branch must keep raising the same way."""

    X = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises((ValueError, TypeError)):
        TransformToNumerical().fit(X)


def test_frame_without_datetime_or_text_is_unaffected():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    tfm = TransformToNumerical().fit(df)
    assert [name for name, _, _ in tfm.tfm_.transformers_] == ["categorical", "continuous"]
    assert tfm.transform(df).shape == (3, 2)
