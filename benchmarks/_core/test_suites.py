"""Tests for suite identity.

Two suites can hold datasets with the same NAME that are different datasets.
``steel-plates-fault`` is OpenML 1504 (binary, trivially separable) in ``real_small`` and
40982 (7-class) in ``cc18_narrow``. If rows from both carried the same suite label, the
aggregator would average them together without any error. These tests pin that it cannot.
"""

from __future__ import annotations

from benchmarks._core.datasets import CC18_NARROW, CC18_WIDE_IDS, REAL_SMALL, SUITES
from benchmarks.suites.real_small import make_tasks


def test_same_name_different_dataset_really_exists():
    """The hazard is real, not hypothetical -- if this ever stops being true, fine."""

    small = {s.name: s.openml_id for s in REAL_SMALL}
    cc18 = {s.name: s.openml_id for s in CC18_NARROW}
    assert small["steel-plates-fault"] != cc18["steel-plates-fault"]


def test_tasks_are_labelled_with_the_dataset_suite_not_the_runner():
    small = make_tasks(REAL_SMALL[:1], 1, 100, "cfg", 0, {}, None, suite="real_small")
    cc18 = make_tasks(CC18_NARROW[:1], 1, 100, "cfg", 0, {}, None, suite="cc18_narrow")
    assert small[0].suite == "real_small"
    assert cc18[0].suite == "cc18_narrow"


def test_colliding_names_get_distinct_resume_keys():
    """Resume keys include the suite, so the two steel-plates never mask each other."""

    small = [s for s in REAL_SMALL if s.name == "steel-plates-fault"]
    cc18 = [s for s in CC18_NARROW if s.name == "steel-plates-fault"]
    t_small = make_tasks(small, 1, 100, "cfg", 0, {}, None, suite="real_small")[0]
    t_cc18 = make_tasks(cc18, 1, 100, "cfg", 0, {}, None, suite="cc18_narrow")[0]
    assert (t_small.suite, t_small.dataset) != (t_cc18.suite, t_cc18.dataset)


def test_rows_record_the_openml_id():
    """The unambiguous identity, for anyone reading the ledger later."""

    task = make_tasks(CC18_NARROW[:1], 1, 100, "cfg", 0, {}, None, suite="cc18_narrow")[0]
    assert task.config["openml_id"] == CC18_NARROW[0].openml_id


def test_cc18_narrow_is_the_mechanical_rule_applied_to_all_72():
    """62 kept + 10 wide = the whole of OpenML-CC18, with no hand-picked removals."""

    assert len(CC18_NARROW) == 62
    assert len(CC18_WIDE_IDS) == 10
    kept = {s.openml_id for s in CC18_NARROW}
    assert kept.isdisjoint(CC18_WIDE_IDS)
    assert len(kept | set(CC18_WIDE_IDS)) == 72


def test_cc18_ids_are_unique():
    ids = [s.openml_id for s in CC18_NARROW]
    assert len(ids) == len(set(ids))


def test_both_suites_are_registered():
    assert SUITES["real_small"] is REAL_SMALL
    assert SUITES["cc18_narrow"] is CC18_NARROW
