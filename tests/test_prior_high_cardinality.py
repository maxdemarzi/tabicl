"""Tests for TP-08: high-cardinality categoricals in the synthetic prior.

The prior drew categorical cardinality from `gammavariate(1, 10)` -- mean 10, thin tail --
so a column with hundreds of levels was essentially never generated. That is the regime the
cell encoder is supposed to resolve, and the TabPFN-3.5 report ties its prior change to its
encoding change explicitly. `high_card_prob` mixes in a log-uniform heavy tail.

It must default to off: the control arm of a prior ablation has to be bit-identical.
"""

from __future__ import annotations

import random
import sys
import types

import pytest
import torch

# The prior package imports xgboost at load time for the v1 TreeSCM.
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBRegressor=object, XGBClassifier=object))

from tabicl.prior._prior_config import DEFAULT_FIXED_HP  # noqa: E402
from tabicl.prior._reg2cls import Reg2Cls  # noqa: E402

N_ROWS = 2000
CEILING = N_ROWS // 4  # the cap _num2cat applies


def _levels(high_card_prob, trials=120, seed=0, **extra):
    """Level counts of converted columns. Unconverted columns stay continuous (~N_ROWS
    distinct), so counting only those at or below the ceiling isolates the converted ones."""

    random.seed(seed)
    torch.manual_seed(seed)
    out = []
    for _ in range(trials):
        r = Reg2Cls.__new__(Reg2Cls)
        r.hp = {"cat_prob": 1.0, "max_categories": float("inf"),
                "high_card_prob": high_card_prob, "min_high_categories": 50, **extra}
        Y = r._num2cat(torch.randn(N_ROWS, 4))
        for c in range(4):
            n = int(torch.unique(Y[:, c]).numel())
            if n <= CEILING:
                out.append(n)
    return out


def test_default_config_leaves_the_prior_unchanged():
    assert DEFAULT_FIXED_HP["high_card_prob"] == 0.0


def test_off_by_default_produces_almost_no_high_cardinality_columns():
    levels = _levels(0.0)
    assert levels
    assert sum(n >= 100 for n in levels) / len(levels) < 0.02


def test_enabling_it_produces_high_cardinality_columns():
    off, on = _levels(0.0), _levels(0.4)
    share = lambda v: sum(n >= 100 for n in v) / len(v)
    assert share(on) > 0.15
    assert share(on) > 10 * share(off) or share(off) == 0


def test_more_probability_means_more_high_cardinality():
    share = lambda v: sum(n >= 100 for n in v) / len(v)
    assert share(_levels(0.5)) > share(_levels(0.2))


def test_cardinality_never_exceeds_what_the_column_can_support():
    """MulticlassAssigner draws boundaries from the data, so levels above the row count
    would yield empty categories rather than a harder problem."""

    assert max(_levels(0.9)) <= CEILING


def test_cardinality_is_always_at_least_two():
    assert min(_levels(0.5)) >= 2


def test_max_categories_still_caps_the_heavy_tail():
    levels = _levels(0.9, max_categories=64)
    assert max(levels) <= 64


def test_identical_seeds_reproduce_identical_prior_data():
    """The control arm must be reproducible, not merely similar."""

    assert _levels(0.0, seed=7) == _levels(0.0, seed=7)
    assert _levels(0.4, seed=7) == _levels(0.4, seed=7)


def test_enabled_prior_differs_from_the_default_one_at_the_same_seed():
    assert _levels(0.4, seed=11) != _levels(0.0, seed=11)
