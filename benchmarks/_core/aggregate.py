"""Aggregation of raw measurements into comparable summaries.

Three summaries, matching how the TabPFN-3.5 report presents its numbers:

- :func:`mean_rank` -- ScoringBench-style. Folds are averaged first, methods are
  ranked within each dataset, and ranks are averaged over datasets.
- :func:`win_rate` -- the report's headline table. Share of splits one method
  wins against another, with datasets weighted equally so a 50-fold dataset does
  not outvote a 3-fold one.
- :func:`elo` -- TabArena/BeyondArena-style. Bradley-Terry fit over pairwise
  per-split comparisons, expressed on the Elo scale.

All functions take a flat sequence of :class:`Record` and are indifferent to how
the measurements were produced, so the same code aggregates a real-data suite, a
synthetic eval set, or a speed sweep.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Record", "records_from_ledger", "mean_rank", "win_rate", "elo", "bootstrap_ci"]


@dataclass(frozen=True)
class Record:
    """One method's score on one split.

    Attributes
    ----------
    method : str
        Identifies what is being compared -- typically ``config_id``, since a
        configuration change is the thing an ablation varies.

    dataset : str
        Dataset name. Aggregation weights datasets equally.

    fold : int
        Split index within the dataset.

    value : float
        The metric value.
    """

    method: str
    dataset: str
    fold: int
    value: float


def records_from_ledger(
    rows: Iterable[dict],
    metric: str,
    suite: Optional[str] = None,
    method_field: str = "config_id",
) -> List[Record]:
    """Project ledger rows onto :class:`Record`.

    Rows carrying an ``error``, or a null value, are dropped -- they are kept in
    the ledger so that gaps stay visible, but they cannot be aggregated.

    Parameters
    ----------
    rows : iterable of dict
        Ledger rows, e.g. from iterating a :class:`~benchmarks._core.schema.Ledger`.

    metric : str
        Which metric to extract. Mixing metrics in one aggregation is meaningless,
        so exactly one must be named.

    suite : str, optional
        Restrict to one suite.

    method_field : str, default="config_id"
        Ledger field identifying the method being compared.

    Returns
    -------
    list of Record
    """

    out: List[Record] = []
    for row in rows:
        if row.get("error") or row.get("value") is None:
            continue
        if row.get("metric") != metric:
            continue
        if suite is not None and row.get("suite") != suite:
            continue
        method = row.get(method_field)
        if method is None:
            continue
        out.append(
            Record(
                method=str(method),
                dataset=str(row.get("dataset")),
                fold=int(row.get("fold", 0)),
                value=float(row["value"]),
            )
        )
    return out


def _per_dataset_means(records: Sequence[Record]) -> Dict[str, Dict[str, float]]:
    """Average folds, giving ``{dataset: {method: mean value}}``."""

    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        acc[rec.dataset][rec.method].append(rec.value)
    return {ds: {m: float(np.mean(v)) for m, v in per_method.items()} for ds, per_method in acc.items()}


def _rank(values: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    """Rank methods, averaging tied ranks. Rank 1 is best."""

    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg
        i = j + 1
    return ranks


def mean_rank(
    records: Sequence[Record],
    higher_is_better: bool = True,
    complete_only: bool = True,
) -> Dict[str, float]:
    """Mean rank across datasets, folds averaged first.

    Parameters
    ----------
    records : sequence of Record

    higher_is_better : bool, default=True
        False for error-like metrics (CRPS, MAE, wall-clock).

    complete_only : bool, default=True
        Restrict to datasets on which every method has a score. Ranking over an
        incomplete grid silently flatters methods that failed on hard datasets,
        which is the most common way to produce a misleading leaderboard. Set
        False only when the gaps are understood.

    Returns
    -------
    dict
        ``{method: mean rank}``, lower is better.
    """

    per_ds = _per_dataset_means(records)
    methods = sorted({rec.method for rec in records})

    if complete_only:
        per_ds = {ds: v for ds, v in per_ds.items() if len(v) == len(methods)}

    acc: Dict[str, List[float]] = defaultdict(list)
    for values in per_ds.values():
        for method, rank in _rank(values, higher_is_better).items():
            acc[method].append(rank)

    return {m: float(np.mean(r)) for m, r in sorted(acc.items())}


def win_rate(
    records: Sequence[Record],
    method: str,
    against: str,
    higher_is_better: bool = True,
    ties_count_half: bool = True,
) -> float:
    """Share of splits ``method`` wins against ``against``, datasets weighted equally.

    Each dataset contributes its own win fraction, and those fractions are then
    averaged -- so a dataset with many folds does not dominate. This is what the
    report means by "datasets weighted equally".

    Returns
    -------
    float
        Win rate in [0, 1], or NaN if the two methods share no split.
    """

    by_split: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for rec in records:
        if rec.method in (method, against):
            by_split[(rec.dataset, rec.fold)][rec.method] = rec.value

    per_ds: Dict[str, List[float]] = defaultdict(list)
    for (dataset, _fold), values in by_split.items():
        if method not in values or against not in values:
            continue
        a, b = values[method], values[against]
        if a == b:
            per_ds[dataset].append(0.5 if ties_count_half else 0.0)
        else:
            won = a > b if higher_is_better else a < b
            per_ds[dataset].append(1.0 if won else 0.0)

    if not per_ds:
        return float("nan")
    return float(np.mean([np.mean(v) for v in per_ds.values()]))


def elo(
    records: Sequence[Record],
    higher_is_better: bool = True,
    anchor: float = 1000.0,
    anchor_method: Optional[str] = None,
    scale: float = 400.0,
    prior_games: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """Bradley-Terry strengths on the Elo scale.

    Pairwise comparisons are formed within each split, then a Bradley-Terry model
    is fit by MM iteration. With :math:`p_i = 10^{R_i/400}` the BT win probability
    :math:`p_i/(p_i+p_j)` is exactly the Elo expected score, so the fitted
    strengths convert to Elo directly.

    Parameters
    ----------
    anchor : float, default=1000.0
        Elo assigned to the anchor method.

    anchor_method : str, optional
        Method pinned to ``anchor``. Defaults to the weakest, matching the
        report's "anchored at 1000" convention.

    prior_games : float, default=1.0
        Virtual tie against an average opponent, added per method. Without it an
        undefeated or winless method has infinite strength and the fit does not
        converge. Small but non-zero by default.

    Returns
    -------
    dict
        ``{method: elo}``, higher is better regardless of ``higher_is_better``.
    """

    methods = sorted({rec.method for rec in records})
    if not methods:
        return {}
    index = {m: i for i, m in enumerate(methods)}
    n = len(methods)

    wins = np.zeros((n, n), dtype=float)  # wins[i, j] = times i beat j

    by_split: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for rec in records:
        by_split[(rec.dataset, rec.fold)][rec.method] = rec.value

    # Weight each dataset equally: a split counts 1 / (folds in its dataset).
    folds_per_dataset: Dict[str, int] = defaultdict(int)
    for dataset, _fold in by_split:
        folds_per_dataset[dataset] += 1

    for (dataset, _fold), values in by_split.items():
        weight = 1.0 / folds_per_dataset[dataset]
        present = sorted(values)
        for a_i in range(len(present)):
            for b_i in range(a_i + 1, len(present)):
                a, b = present[a_i], present[b_i]
                va, vb = values[a], values[b]
                ia, ib = index[a], index[b]
                if va == vb:
                    wins[ia, ib] += 0.5 * weight
                    wins[ib, ia] += 0.5 * weight
                elif (va > vb) if higher_is_better else (va < vb):
                    wins[ia, ib] += weight
                else:
                    wins[ib, ia] += weight

    if prior_games > 0:
        # Half a virtual win and half a virtual loss against every other method.
        off_diag = ~np.eye(n, dtype=bool)
        wins[off_diag] += prior_games / (2.0 * max(n - 1, 1))

    total_wins = wins.sum(axis=1)
    games = wins + wins.T

    strength = np.ones(n, dtype=float)
    for _ in range(max_iter):
        prev = strength.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or games[i, j] == 0:
                    continue
                denom += games[i, j] / (strength[i] + strength[j])
            if denom > 0 and total_wins[i] > 0:
                strength[i] = total_wins[i] / denom
        strength = np.maximum(strength, 1e-12)
        strength /= float(np.exp(np.mean(np.log(strength))))  # geometric mean 1
        if np.max(np.abs(strength - prev)) < tol:
            break

    ratings = {m: scale * math.log10(strength[index[m]]) for m in methods}

    pin = anchor_method if anchor_method in ratings else min(ratings, key=ratings.get)
    shift = anchor - ratings[pin]
    return {m: r + shift for m, r in sorted(ratings.items())}


def bootstrap_ci(
    records: Sequence[Record],
    statistic: Callable[[Sequence[Record]], Dict[str, float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap CIs, resampling *datasets* with replacement.

    Datasets are the unit of resampling, not individual splits: folds within a
    dataset are correlated, so resampling splits would understate uncertainty.

    Parameters
    ----------
    statistic : callable
        Any of :func:`mean_rank`, :func:`elo`, or a partial thereof. Applied to
        each resample.

    Returns
    -------
    dict
        ``{method: (low, high)}``. Methods absent from a resample are skipped for
        that draw rather than imputed.
    """

    by_dataset: Dict[str, List[Record]] = defaultdict(list)
    for rec in records:
        by_dataset[rec.dataset].append(rec)
    datasets = sorted(by_dataset)
    if not datasets:
        return {}

    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_boot):
        picked = rng.choice(len(datasets), size=len(datasets), replace=True)
        resample: List[Record] = []
        for k, idx in enumerate(picked):
            # Rename duplicates so a dataset drawn twice counts twice.
            for rec in by_dataset[datasets[idx]]:
                resample.append(Record(rec.method, f"{rec.dataset}#{k}", rec.fold, rec.value))
        for method, value in statistic(resample).items():
            draws[method].append(value)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: (float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))) for m, v in sorted(draws.items()) if v}
