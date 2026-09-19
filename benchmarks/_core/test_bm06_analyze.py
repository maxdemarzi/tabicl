"""Pin the BM-06 verdict to ledgers whose answer is known.

This script's output decides whether Phase 2 ablations get screened at proxy scale, which is
the bulk of the remaining GPU budget. So each verdict is tested against a synthetic ledger
built with a known effect -- including the false-positive trap the first version fell into,
where a chance p < 0.05 at the first checkpoint was reported as the onset of a real effect.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

from benchmarks.bm06_analyze import main


def _ledger(tmp_path, gaps, seed=0, n_datasets=40):
    """gaps: {step: log-loss added to the treatment arm}. Positive = treatment worse."""

    rng = np.random.default_rng(seed)
    path = tmp_path / f"l{seed}.jsonl"
    with path.open("w") as f:
        for step, gap in gaps.items():
            for d in range(n_datasets):
                base = 0.5 + rng.normal(0, 0.1)
                for fold in range(3):
                    for arm, extra in (("control", 0.0), ("adamw", gap)):
                        ll = base + extra + rng.normal(0, 0.01)
                        for metric, val in (("log_loss", ll), ("accuracy", 1 - ll / 2)):
                            f.write(json.dumps({
                                "run_id": "t", "suite": "s", "dataset": f"d{d}", "fold": fold,
                                "config_id": f"{arm}-step{step}", "metric": metric,
                                "value": val, "error": None}) + "\n")
    return path


def _verdict(path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        main([str(path)])
    return next(l for l in buf.getvalue().splitlines() if l.startswith("VERDICT"))


def test_reports_the_true_onset_not_an_earlier_chance_hit(tmp_path):
    """The bug the first version had: no effect at 2500, real effect from 5000."""

    v = _verdict(_ledger(tmp_path, {2500: 0.0, 5000: 0.04, 7500: 0.04}, seed=0))
    assert "from step 5000" in v


def test_no_effect_reads_as_no_separation(tmp_path):
    v = _verdict(_ledger(tmp_path, {s: 0.0 for s in (2500, 5000, 7500, 10000)}, seed=1))
    assert "no separation" in v


def test_opposite_effect_is_reported_as_a_contradiction_not_as_nothing(tmp_path):
    v = _verdict(_ledger(tmp_path, {2500: -0.04, 5000: -0.04}, seed=2))
    assert "OPPOSITE" in v


def test_a_sign_that_does_not_persist_is_not_trusted(tmp_path):
    v = _verdict(_ledger(tmp_path, {2500: 0.04, 5000: 0.0, 7500: 0.04}, seed=3))
    assert "does not hold" in v


def test_false_positive_rate_on_pure_noise_is_controlled(tmp_path):
    """Eight checkpoints of pure noise must rarely yield a 'reproduces' verdict.

    Uncorrected, the chance of at least one p < 0.05 across 8 checkpoints is ~34%.
    """

    steps = (2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)
    hits = sum("reproduces" in _verdict(_ledger(tmp_path, {s: 0.0 for s in steps}, seed=100 + i))
               for i in range(25))
    assert hits <= 2


def test_too_few_shared_datasets_does_not_produce_a_verdict_row(tmp_path):
    path = _ledger(tmp_path, {2500: 0.04}, seed=4, n_datasets=3)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main([str(path)])
    assert "too few datasets" in buf.getvalue()


def test_a_single_checkpoint_is_only_provisional(tmp_path):
    """Persistence is undefined with one checkpoint; the verdict must not claim it."""

    v = _verdict(_ledger(tmp_path, {2500: 0.04}, seed=5))
    assert "PROVISIONAL" in v and "onward" not in v
