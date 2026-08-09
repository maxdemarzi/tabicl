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

__all__ = ["LeakageReport", "permutation_control", "permutation_test", "temporal_control"]


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

    inconclusive : bool
        The control could not render a verdict. Distinct from ``passed``: a pass is
        evidence of no leak, this is an absence of evidence either way. Callers must not
        read it as a pass *or* as a failure -- see `temporal_control`.
    """

    passed: bool
    observed: float
    control: List[float] = field(default_factory=list)
    chance: float = 0.5
    reason: str = ""
    inconclusive: bool = False

    @property
    def control_mean(self) -> float:
        return float(np.mean(self.control)) if self.control else float("nan")

    def __repr__(self) -> str:  # pragma: no cover - display only
        verdict = ("INCONCLUSIVE" if self.inconclusive
                   else "PASS" if self.passed else "*** LEAK ***")
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
    # TWO-SIDED, on |mean - chance|. The test asks whether features rebuilt from SHUFFLED
    # labels still carry information about the true one, and information is distance from
    # chance in either direction: a permuted-label score of 0.35 predicts the true label
    # exactly as well as 0.65 does, inverted.
    #
    # The one-sided form missed that entire half. rel-avito/user-clicks passed this control
    # at 0.4415 +- 0.0078 -- 7.5 sd BELOW chance -- while `mean <= chance + tolerance` read
    # 0.4415 <= 0.52 and reported PASS. Its `+rate`, `+history` and `+text+rate` arms were
    # eligible on that basis.
    #
    # Note the direction of this change relative to the temporal control fixed alongside it.
    # The same principle -- compare information, not raw score -- makes THAT control more
    # permissive and this one STRICTER. A principle that only ever loosened the gates in our
    # favour would be worth distrusting.
    deviation = abs(mean - chance)
    passed = deviation <= tolerance
    reason = (
        f"permuted-label score {mean:.4f} is within {tolerance} of chance {chance}"
        if passed else
        f"permuted labels still carry information: {mean:.4f} is {deviation:.4f} from "
        f"chance {chance} -- the feature is reading the row's own label"
        + (" (inverted, which the one-sided test missed)" if mean < chance else "")
    )
    return LeakageReport(passed, observed, control, chance, reason)


def permutation_test(
    build_and_score: Callable[[np.ndarray], float],
    y: np.ndarray,
    n_permutations: int = 5,
    n_sigma: float = 3.0,
    random_state: Optional[int] = 0,
) -> LeakageReport:
    """Permutation with the *empirical* null instead of chance.

    Use this, not `permutation_control`, whenever the feature encodes structure as well as
    labels -- which is most graph features. `permutation_control` asks whether the permuted
    score is near 0.5, and that is only the right question when the feature's entire content
    is labels.

    A neighbour positive-rate is not such a feature. Its *noise* carries degree: a query with
    one labelled neighbour scores 0 or 1, a query with twenty scores near the mean. Degree is
    label-independent, survives any permutation, and on rel-event predicts the target on its
    own at AUC 73.2 -- so the permuted score lands at 52.4, and a chance-based threshold calls
    a sound feature a leak.

    The right question is whether the real labels beat their own permuted null. Note this test
    cannot detect a row reading its own label; it is not a substitute for `permutation_control`
    where that control applies, nor for arranging the computation so the leak is impossible.

    Parameters
    ----------
    build_and_score : callable
        ``build_and_score(y) -> float``, rebuilding the features from the labels it is given.

    y : np.ndarray
        The real labels.

    n_permutations : int, default=5
        Shuffles forming the null. More than `permutation_control` needs, because here the
        null's *spread* is used and not just its location.

    n_sigma : float, default=3.0
        How far above the null's mean the observed score must sit.

    random_state : int, optional
        Seed for the shuffles.

    Returns
    -------
    LeakageReport
        ``chance`` carries the null's mean rather than 0.5.
    """
    rng = np.random.default_rng(random_state)
    observed = float(build_and_score(y))
    control = [float(build_and_score(rng.permutation(y))) for _ in range(n_permutations)]

    mean = float(np.mean(control))
    sd = float(np.std(control, ddof=1)) if len(control) > 1 else 0.0
    threshold = mean + n_sigma * sd
    passed = observed > threshold
    reason = (
        f"observed {observed:.4f} beats its permuted null {mean:.4f} +- {sd:.4f} "
        f"({(observed - mean) / sd:.1f} sd)" if passed and sd > 0 else
        f"observed {observed:.4f} beats its permuted null {mean:.4f}" if passed else
        f"observed {observed:.4f} does not clear the permuted null {mean:.4f} +- {sd:.4f} "
        f"at {n_sigma} sd -- the label content adds nothing beyond structure"
    )
    return LeakageReport(passed, observed, control, mean, reason)


def temporal_control(
    score_at_cutoff_shift: Callable[[float], float],
    shifts: Sequence[float] = (0.0, 30.0, 90.0),
    tolerance: float = 0.005,
    chance: float = 0.5,
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
    Withholding history cannot *add* information, so an increase means the features were
    reaching past the cutoff. This catches the failure the permutation control cannot: a
    feature that uses only other rows' labels, correctly, but reads them from the future.

    "Information" is ``|score - chance|``, not the raw score. These controls score a single
    column directly against the label with no model fitted, so a value below chance is an
    *inverted* feature rather than a weak one -- 0.32 carries what 0.68 carries, and a model
    uses it by learning the negative sign. Above chance the two measures agree exactly.
    """
    if not shifts or shifts[0] != 0.0:
        raise ValueError("shifts must start at 0.0, the unshifted setting")

    observed = float(score_at_cutoff_shift(shifts[0]))
    control = [float(score_at_cutoff_shift(s)) for s in shifts[1:]]

    # Compare INFORMATION, |score - chance|, not the raw score.
    #
    # These controls score a single summed column directly against the label -- no model is
    # fitted -- so a value below chance is not a weak feature but an inverted one. On
    # rel-amazon, items with more reviews are LESS likely to churn: `n_linked` scores 0.32,
    # which carries exactly the information of 0.68 and which any model uses by learning the
    # negative sign.
    #
    # Comparing raw scores therefore inverts the verdict in that regime. rel-amazon/
    # item-churn's count block read 0.3236 unshifted and 0.3388 shifted, so the raw test
    # reported *** LEAK *** -- while the deviations are 0.176 and 0.161, meaning withholding
    # history REDUCED the information, which is the opposite. Six such verdicts were issued
    # across this project's runs against seven genuine ones, excluding five of seven arms on
    # both rel-amazon tasks and `+struct` on rel-event/user-repeat and rel-trial/
    # study-outcome, our two best results.
    #
    # Above chance the two measures agree exactly, so every genuine verdict is unchanged.
    # An AUC outside [0, 1] means the caller's scoring function is broken, and reading it
    # as extreme information would be the worst response. A test in this project asserted a
    # pass on an AUC of -0.10 for months because the raw comparison accepted it silently.
    if chance == 0.5:
        bad = [s for s in (observed, *control) if not (0.0 <= s <= 1.0)]
        if bad:
            raise ValueError(
                f"score(s) {bad} lie outside [0, 1] and cannot be ROC-AUC; "
                f"temporal_control compares |score - chance| and cannot interpret them")

    info_obs = abs(observed - chance)
    infos = [abs(c - chance) for c in control]
    worst = max(control, key=lambda c: abs(c - chance)) if control else observed
    info_worst = max(infos) if infos else info_obs

    # Both sides indistinguishable from chance: there is no information on either side to
    # compare, so the control has nothing to say. Absence of evidence, reported as such.
    if info_obs <= tolerance and info_worst <= tolerance:
        return LeakageReport(
            True, observed, control, chance,
            f"cannot judge: the block scores {observed:.4f} against chance {chance:.2f} and "
            f"the shifted runs {worst:.4f} -- both within {tolerance} of chance, so there is "
            f"no information on either side to compare",
            inconclusive=True)

    passed = info_worst <= info_obs + tolerance
    reason = (
        f"earlier cutoffs do not add information to {observed:.4f} "
        f"(|obs-chance| {info_obs:.4f} vs best shifted {info_worst:.4f})"
        if passed else
        f"withholding history *increased* the information from |{observed:.4f}-chance| = "
        f"{info_obs:.4f} to |{worst:.4f}-chance| = {info_worst:.4f} -- the features are "
        f"reaching past the cutoff"
    )
    return LeakageReport(passed, observed, control, chance, reason)
