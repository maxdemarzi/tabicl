"""Negative controls for label-derived features.

`RESEARCH.md` 6b and 6c propose features built from *neighbours' labels* -- k-hop
positive counts, label-typed triangles. They are the direct attack on the i.i.d. limit
that caps flattening, and they are also the highest-risk family this package has
attempted: a leak there produces a large, confident, entirely fake number rather than an
error.

That risk is not hypothetical here. Several conclusions in `DESIGN.md` were wrong for
exactly that reason -- they produced plausible numbers, not exceptions. A `-3.21` stood as
a finding for hours; an AMP default cost 7.3 AUC while every run looked healthy. So the
controls come before the features.

Two controls, testing different failure modes:

**Permutation.** Shuffle the labels, rebuild the features from the shuffled labels, score.
A correctly built label feature carries no signal about a randomised target, so the score
must collapse to chance. If it does not, the feature is reading the row's own label --
through a self-loop, a 0-hop term, or a target column that survived into the matrix.

**Temporal.** Rebuild with every cutoff moved earlier. A causal feature can only lose
information when it is shown less history, so scores must not *improve*. An improvement
means the "history" was reaching forward.

Neither proves absence of leakage. Both catch the mistakes that are actually made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = ["LeakageReport", "permutation_control", "temporal_control"]


@dataclass
class LeakageReport:
    """Outcome of a negative control.

    Attributes
    ----------
    passed : bool
        Whether the control found no evidence of leakage. False is a hard stop: a feature
        that fails this should not be measured, let alone reported.

    observed : float
        Score on the real labels or the real cutoffs.

    control : list of float
        Scores under the control condition.

    chance : float
        The score a signal-free feature should produce (0.5 for ROC-AUC).

    reason : str
        Human-readable verdict, suitable for printing next to a result.
    """

    passed: bool
    observed: float
    control: List[float] = field(default_factory=list)
    chance: float = 0.5
    reason: str = ""

    @property
    def control_mean(self) -> float:
        return float(np.mean(self.control)) if self.control else float("nan")

    def __repr__(self) -> str:  # pragma: no cover - display only
        verdict = "PASS" if self.passed else "*** LEAK ***"
        return (f"LeakageReport({verdict}, observed={self.observed:.4f}, "
                f"control={self.control_mean:.4f}, {self.reason})")


def permutation_control(
    build_and_score: Callable[[np.ndarray], float],
    y: np.ndarray,
    n_permutations: int = 5,
    chance: float = 0.5,
    tolerance: float = 0.02,
    random_state: Optional[int] = 0,
) -> LeakageReport:
    """Shuffle the labels, rebuild the features from them, and score.

    Parameters
    ----------
    build_and_score : callable
        ``build_and_score(y) -> float``. Must rebuild the label-derived features **from
        the labels it is given** and return a score where higher is better. Passing a
        function that closes over the real labels defeats the entire control, which is
        the one mistake to guard against when wiring this up.

    y : np.ndarray
        The real labels.

    n_permutations : int, default=5
        Shuffles to average over. Small because a leak is not subtle -- it shows up as a
        control score near the observed one, not as a fractional shift.

    chance : float, default=0.5
        Score expected from a signal-free feature. 0.5 for ROC-AUC.

    tolerance : float, default=0.02
        How far above ``chance`` the control may land before it is called a leak.

    random_state : int, optional
        Seed for the shuffles.

    Returns
    -------
    LeakageReport
    """
    rng = np.random.default_rng(random_state)
    observed = float(build_and_score(y))
    control = [float(build_and_score(rng.permutation(y))) for _ in range(n_permutations)]

    mean = float(np.mean(control))
    passed = mean <= chance + tolerance
    reason = (
        f"permuted-label score {mean:.4f} is within {tolerance} of chance {chance}"
        if passed else
        f"permuted labels still score {mean:.4f} against chance {chance} -- the feature "
        f"is reading the row's own label"
    )
    return LeakageReport(passed, observed, control, chance, reason)


def temporal_control(
    score_at_cutoff_shift: Callable[[float], float],
    shifts: Sequence[float] = (0.0, 30.0, 90.0),
    tolerance: float = 0.005,
) -> LeakageReport:
    """Move every cutoff earlier and confirm scores do not improve.

    Parameters
    ----------
    score_at_cutoff_shift : callable
        ``score_at_cutoff_shift(days_earlier) -> float``. Shift 0 is the real setting.

    shifts : Sequence[float], default=(0.0, 30.0, 90.0)
        Days to move cutoffs earlier. Must start at 0.

    tolerance : float, default=0.005
        How much a shifted score may exceed the unshifted one before it is called a leak.
        Small but non-zero, since shifting also changes which rows have any history at
        all and that can help a little by chance.

    Returns
    -------
    LeakageReport

    Notes
    -----
    Withholding history cannot *add* information, so an improvement means the features
    were reaching past the cutoff. This catches the failure the permutation control
    cannot: a feature that uses only other rows' labels, correctly, but reads them from
    the future.
    """
    if not shifts or shifts[0] != 0.0:
        raise ValueError("shifts must start at 0.0, the unshifted setting")

    observed = float(score_at_cutoff_shift(shifts[0]))
    control = [float(score_at_cutoff_shift(s)) for s in shifts[1:]]
    worst = max(control) if control else observed

    passed = worst <= observed + tolerance
    reason = (
        f"earlier cutoffs do not improve on {observed:.4f} (best {worst:.4f})"
        if passed else
        f"withholding history *improved* the score to {worst:.4f} from {observed:.4f} -- "
        f"the features are reaching past the cutoff"
    )
    return LeakageReport(passed, observed, control, observed, reason)
