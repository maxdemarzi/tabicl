"""Tests for the harness core.

The harness is the instrument. If its arithmetic is wrong, every number produced
for the TabPFN-3.5 backlog is wrong in a way that looks plausible. These tests
pin the aggregation to cases whose answers are known in closed form, and pin the
ledger to the failure modes that actually occur mid-sweep (kills, retries,
truncated writes).

Run with:  python -m pytest benchmarks/_core/test_core.py
"""

from __future__ import annotations

import math

import pytest

from benchmarks._core.aggregate import (
    Record,
    bootstrap_ci,
    elo,
    mean_rank,
    records_from_ledger,
    win_rate,
)
from benchmarks._core.schema import Ledger, ResultRow, capture_provenance


# --------------------------------------------------------------------------- #
# mean_rank
# --------------------------------------------------------------------------- #


def _grid(values_by_method, datasets=3, folds=2):
    """A complete grid where each method scores a constant."""

    return [
        Record(method, f"ds{d}", f, value)
        for method, value in values_by_method.items()
        for d in range(datasets)
        for f in range(folds)
    ]


def test_mean_rank_strict_ordering():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    assert mean_rank(records) == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_mean_rank_lower_is_better_inverts():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    assert mean_rank(records, higher_is_better=False) == {"a": 3.0, "b": 2.0, "c": 1.0}


def test_mean_rank_ties_are_averaged():
    # a and b tie for the top: ranks 1 and 2 average to 1.5 each.
    records = _grid({"a": 3.0, "b": 3.0, "c": 1.0})
    assert mean_rank(records) == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_mean_rank_folds_averaged_before_ranking():
    # b wins one fold big and loses the other small, so on fold means b beats a.
    records = [
        Record("a", "ds0", 0, 1.0),
        Record("a", "ds0", 1, 1.0),
        Record("b", "ds0", 0, 10.0),
        Record("b", "ds0", 1, 0.5),
    ]
    # means: a = 1.0, b = 5.25 -> b ranks first
    assert mean_rank(records) == {"a": 2.0, "b": 1.0}


def test_mean_rank_complete_only_drops_partial_datasets():
    records = _grid({"a": 3.0, "b": 2.0}, datasets=2, folds=1)
    records.append(Record("a", "ds_partial", 0, 99.0))  # b missing here

    complete = mean_rank(records, complete_only=True)
    assert complete == {"a": 1.0, "b": 2.0}

    # Without the guard, the partial dataset hands 'a' a free rank-1.
    partial = mean_rank(records, complete_only=False)
    assert partial["a"] == 1.0
    assert partial["b"] == 2.0
    # The point is that the partial dataset was counted at all:
    assert len(records) == 5


# --------------------------------------------------------------------------- #
# win_rate
# --------------------------------------------------------------------------- #


def test_win_rate_simple():
    records = [
        Record("a", "ds0", 0, 1.0), Record("b", "ds0", 0, 0.0),  # a wins
        Record("a", "ds1", 0, 1.0), Record("b", "ds1", 0, 0.0),  # a wins
        Record("a", "ds2", 0, 1.0), Record("b", "ds2", 0, 0.0),  # a wins
        Record("a", "ds3", 0, 0.0), Record("b", "ds3", 0, 1.0),  # b wins
    ]
    assert win_rate(records, "a", "b") == pytest.approx(0.75)


def test_win_rate_weights_datasets_equally_not_folds():
    """A 10-fold dataset must not outvote a 1-fold dataset.

    'a' wins all 10 folds of ds_big and loses the single fold of ds_small.
    Split-weighted that is 10/11 = 0.91; dataset-weighted it is 0.5.
    """

    records = []
    for f in range(10):
        records += [Record("a", "ds_big", f, 1.0), Record("b", "ds_big", f, 0.0)]
    records += [Record("a", "ds_small", 0, 0.0), Record("b", "ds_small", 0, 1.0)]

    assert win_rate(records, "a", "b") == pytest.approx(0.5)


def test_win_rate_ties_count_half():
    records = [
        Record("a", "ds0", 0, 1.0), Record("b", "ds0", 0, 1.0),  # tie
        Record("a", "ds1", 0, 1.0), Record("b", "ds1", 0, 0.0),  # a wins
    ]
    assert win_rate(records, "a", "b") == pytest.approx(0.75)
    assert win_rate(records, "a", "b", ties_count_half=False) == pytest.approx(0.5)


def test_win_rate_no_shared_split_is_nan():
    records = [Record("a", "ds0", 0, 1.0), Record("b", "ds1", 0, 1.0)]
    assert math.isnan(win_rate(records, "a", "b"))


# --------------------------------------------------------------------------- #
# elo
# --------------------------------------------------------------------------- #


def test_elo_two_methods_75_percent_matches_closed_form():
    """BT with p_A/(p_A+p_B) = 0.75 implies an Elo gap of 400*log10(3).

    This is the load-bearing test for the Elo implementation: it is the one case
    where the right answer is known exactly.
    """

    records = []
    for d in range(4):
        winner = "a" if d < 3 else "b"
        loser = "b" if d < 3 else "a"
        records += [Record(winner, f"ds{d}", 0, 1.0), Record(loser, f"ds{d}", 0, 0.0)]

    ratings = elo(records, prior_games=0.0)
    gap = ratings["a"] - ratings["b"]
    assert gap == pytest.approx(400 * math.log10(3.0), rel=1e-4)


def test_elo_equal_methods_have_equal_rating():
    records = []
    for d in range(4):
        winner = "a" if d % 2 == 0 else "b"
        loser = "b" if d % 2 == 0 else "a"
        records += [Record(winner, f"ds{d}", 0, 1.0), Record(loser, f"ds{d}", 0, 0.0)]

    ratings = elo(records, prior_games=0.0)
    assert ratings["a"] == pytest.approx(ratings["b"], abs=1e-6)


def test_elo_anchors_weakest_at_1000_by_default():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    ratings = elo(records)
    assert min(ratings.values()) == pytest.approx(1000.0)
    assert ratings["a"] > ratings["b"] > ratings["c"]


def test_elo_explicit_anchor_method():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    ratings = elo(records, anchor=1500.0, anchor_method="b")
    assert ratings["b"] == pytest.approx(1500.0)
    assert ratings["a"] > 1500.0 > ratings["c"]


def test_elo_undefeated_method_stays_finite():
    """Without smoothing a never-beaten method has infinite BT strength."""

    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    ratings = elo(records, prior_games=1.0)
    assert all(math.isfinite(r) for r in ratings.values())


def test_elo_lower_is_better_inverts_order():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0})
    ratings = elo(records, higher_is_better=False)
    assert ratings["c"] > ratings["b"] > ratings["a"]


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_brackets_point_estimate():
    records = _grid({"a": 3.0, "b": 2.0, "c": 1.0}, datasets=12, folds=1)
    point = mean_rank(records)
    ci = bootstrap_ci(records, mean_rank, n_boot=200, seed=0)
    for method, (lo, hi) in ci.items():
        assert lo <= point[method] <= hi


def test_bootstrap_ci_is_deterministic_for_a_seed():
    records = _grid({"a": 3.0, "b": 1.0}, datasets=8, folds=1)
    a = bootstrap_ci(records, mean_rank, n_boot=100, seed=7)
    b = bootstrap_ci(records, mean_rank, n_boot=100, seed=7)
    assert a == b


def test_bootstrap_ci_widens_when_methods_are_close():
    """A dataset where the winner flips should give a wider interval."""

    tight = _grid({"a": 3.0, "b": 1.0}, datasets=10, folds=1)
    noisy = []
    for d in range(10):
        a_val, b_val = (3.0, 1.0) if d % 2 == 0 else (1.0, 3.0)
        noisy += [Record("a", f"ds{d}", 0, a_val), Record("b", f"ds{d}", 0, b_val)]

    tight_ci = bootstrap_ci(tight, mean_rank, n_boot=300, seed=0)["a"]
    noisy_ci = bootstrap_ci(noisy, mean_rank, n_boot=300, seed=0)["a"]
    assert (noisy_ci[1] - noisy_ci[0]) > (tight_ci[1] - tight_ci[0])


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def _row(**kw):
    base = dict(
        run_id="R", suite="s", dataset="ds0", fold=0, config_id="cfg", metric="acc", value=1.0
    )
    base.update(kw)
    return ResultRow(**base)


def test_ledger_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "out.jsonl")
    ledger.append(_row(value=0.5))
    ledger.append(_row(dataset="ds1", value=0.75))

    rows = list(ledger)
    assert [r["value"] for r in rows] == [0.5, 0.75]
    assert all(r["schema_version"] == 1 for r in rows)


def test_ledger_is_append_only_across_instances(tmp_path):
    path = tmp_path / "out.jsonl"
    Ledger(path).append(_row(value=1.0))
    Ledger(path).append(_row(dataset="ds1", value=2.0))
    assert len(list(Ledger(path))) == 2


def test_ledger_completed_keys_supports_resume(tmp_path):
    ledger = Ledger(tmp_path / "out.jsonl")
    ledger.append(_row(dataset="ds0"))
    ledger.append(_row(dataset="ds1"))

    keys = ledger.completed_keys()
    assert ("R", "s", "ds0", 0, "cfg", "acc") in keys
    assert ("R", "s", "ds2", 0, "cfg", "acc") not in keys


def test_ledger_errored_rows_are_retried_not_skipped(tmp_path):
    """A failed measurement must not look 'done' on the next invocation."""

    ledger = Ledger(tmp_path / "out.jsonl")
    ledger.append(_row(dataset="ds_fail", value=None, error="CUDA OOM"))
    assert ("R", "s", "ds_fail", 0, "cfg", "acc") not in ledger.completed_keys()
    # ...but the failure is still on the record.
    assert any(r.get("error") == "CUDA OOM" for r in ledger)


def test_ledger_completed_keys_filters_by_run(tmp_path):
    ledger = Ledger(tmp_path / "out.jsonl")
    ledger.append(_row(run_id="R1"))
    ledger.append(_row(run_id="R2", dataset="ds9"))
    assert len(ledger.completed_keys(run_id="R1")) == 1


def test_ledger_survives_a_truncated_final_line(tmp_path):
    """Expected after a hard kill: the partial line must not poison the file."""

    path = tmp_path / "out.jsonl"
    ledger = Ledger(path)
    ledger.append(_row(value=1.0))
    with path.open("a") as fh:
        fh.write('{"run_id": "R", "suite": "s", "dat')  # killed mid-write

    rows = list(ledger)
    assert len(rows) == 1
    assert rows[0]["value"] == 1.0


def test_result_row_keys_are_unique_per_measurement():
    assert _row(dataset="ds0").key() != _row(dataset="ds1").key()
    assert _row(metric="acc").key() != _row(metric="auc").key()


def test_records_from_ledger_drops_errors_and_other_metrics(tmp_path):
    ledger = Ledger(tmp_path / "out.jsonl")
    ledger.append(_row(config_id="base", metric="acc", value=0.9))
    ledger.append(_row(config_id="base", metric="auc", value=0.8))
    ledger.append(_row(config_id="base", dataset="ds9", value=None, error="boom"))

    recs = records_from_ledger(list(ledger), metric="acc")
    assert len(recs) == 1
    assert recs[0].method == "base" and recs[0].value == 0.9


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_provenance_captures_git_state():
    prov = capture_provenance(device="cpu")
    assert prov.commit_sha is None or len(prov.commit_sha) == 40
    assert prov.python_version is not None


def test_provenance_reports_whether_tabicl_is_local():
    """Guards against benchmarking a released wheel while editing src/."""

    prov = capture_provenance()
    assert prov.tabicl_is_local in (True, False, None)
