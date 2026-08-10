"""The headline table, computed rather than hand-maintained.

This table is where this project's transcription errors have happened: an average-rank cell
that was a mean of ranks rather than a rank of means, rank claims that had to be qualified
after the fact, and a `best-cfg` parser that once reported a competitor's number as ours
because the reference suffix was not stripped. All of those are arithmetic over a fixed set
of published numbers, which is work a machine should do.

**The ranking rule was inferred from the seven published cells, not assumed**, and it
reproduces all seven exactly:

    rank = 1 + (how many of the NINE Table-14 methods score above us), out of 10

DFS is displayed in the table but **not counted**, which is what makes the denominator 10
rather than 11. Its numbers come from the RelBench paper rather than Table 14 and it is
missing on several tasks, so counting it would silently change the denominator per task.

Field figures: TabPFN-3 Table 14, ``arXiv 2605.13986`` p.61. KumoRFMv1 and RTzero are
excluded because the report itself flags them as following a different evaluation protocol
that overestimates performance.
"""

from __future__ import annotations

import statistics

# The nine ranked methods, all twelve RelBenchV1 entity classification tasks.
METHODS = ("RelGNN", "RelGT", "GraphSAGE", "Griffin", "RDBLearn",
           "RDBLearn+v2.5", "RDBLearn+v3", "KumoRFMv2", "TabPFN-REL")

FIELD: dict[str, dict[str, float]] = {
    "rel-f1/driver-dnf":       (75.29, 75.87, 72.62, 57.70, 70.87, 71.72, 71.72, 72.03, 70.74),
    "rel-f1/driver-top3":      (85.69, 83.52, 75.54, 82.50, 79.69, 77.60, 82.72, 82.09, 79.98),
    "rel-avito/user-clicks":   (68.23, 68.30, 65.90, 45.90, 69.04, 65.72, 69.06, 67.42, 67.09),
    "rel-avito/user-visits":   (66.18, 66.78, 66.20, 60.70, 65.49, 66.47, 66.76, 69.41, 66.68),
    "rel-event/user-repeat":   (79.61, 76.09, 76.89, 71.88, 75.04, 75.55, 76.81, 79.34, 77.11),
    "rel-event/user-ignore":   (86.18, 81.57, 81.62, 83.27, 82.52, 78.65, 73.70, 78.86, 85.38),
    "rel-trial/study-outcome": (71.24, 68.61, 68.60, 51.00, 71.58, 72.90, 72.89, 72.03, 76.43),
    "rel-amazon/user-churn":   (70.99, 70.39, 70.42, 62.30, 67.57, 69.74, 69.35, 67.71, 70.27),
    "rel-amazon/item-churn":   (82.64, 82.55, 82.81, 69.00, 82.07, 82.18, 82.46, 80.18, 82.81),
    "rel-stack/user-engagement": (90.75, 90.53, 90.59, 77.50, 89.39, 90.23, 90.59, 88.69, 90.66),
    "rel-stack/user-badge":    (88.98, 86.32, 88.86, 73.50, 85.26, 82.81, 85.98, 85.40, 85.17),
    "rel-hm/user-churn":       (70.93, 69.27, 69.88, 60.20, 68.05, 70.11, 70.06, 67.81, 70.55),
}
FIELD = {t: dict(zip(METHODS, v)) for t, v in FIELD.items()}

# Ours, official protocol. Only tasks with a measured number belong here: a task we have not
# run is absent, never zero and never imputed.
OURS: dict[str, float] = {
    "rel-event/user-repeat": 77.89,
    "rel-trial/study-outcome": 72.26,
    "rel-f1/driver-top3": 81.98,
    "rel-event/user-ignore": 80.98,
    "rel-avito/user-visits": 65.54,
    "rel-avito/user-clicks": 65.89,
    "rel-f1/driver-dnf": 69.66,
    # Added 2026-08-08, the first of the five missing tasks. Places 9/10.
    "rel-hm/user-churn": 66.75,
    # Added 2026-08-09. rel-amazon/user-churn is a complete 5-replicate run; user-engagement
    # is FOUR seeds, salvaged from a run the 3h ceiling killed before it printed a summary
    # (~55 min/seed with disk offloading). Both are real measurements and one is weaker than
    # the standard five; PARTIAL_SEEDS records which, so the table can mark it.
    "rel-stack/user-engagement": 89.33,
    # 66.94 was measured under the superseded temporal control, which left this task a
    # single 52-column arm. With seven arms the old configuration no longer runs; two
    # re-measurement attempts gave 67.57 (1 seed, disk-offload I/O error) and 67.22
    # (2 seeds, timeout), both inside the floor and both too thin to publish. Kept, with
    # the caveat recorded in PERFORMANCE.md rather than hidden.
    "rel-amazon/user-churn": 66.94,
    # Added 2026-08-09, the eleventh. Took --train-pool (2.54M -> 300k, so feature
    # construction fit) AND dropping rel-amazon's array-valued column, which crashed
    # pd.factorize after the controls had already run. Its leak controls excluded five of
    # seven arms, so this is a `base`-only result at 55 columns.
    "rel-amazon/item-churn": 80.20,
    # Added 2026-08-09, the twelfth and last. Six attempts: --row-chunk, --offload cpu,
    # --offload disk and --train-pool all bound the model or the fit pool and all failed
    # rc=137. What unblocked it was dropping child rows whose key is never queried, which
    # bounds the AGGREGATION. Four salvaged replicates.
    "rel-stack/user-badge": 83.80,
}

# Tasks measured with fewer than the standard five replicates, and how many they got.
PARTIAL_SEEDS: dict[str, int] = {"rel-stack/user-engagement": 4,
                                 "rel-stack/user-badge": 4}


def rank(score: float, task: str) -> tuple[int, int]:
    """Our rank on one task: ``(rank, field_size + 1)``.

    Ties go to us -- strictly-greater scores count. That is the convention the seven
    published cells were computed under and it is stated here because the alternative is
    defensible too; what matters is that it never changes silently.
    """
    if task not in FIELD:
        raise KeyError(f"no published field for {task}; refusing to invent a rank")
    beat = sum(1 for v in FIELD[task].values() if v > score)
    return beat + 1, len(FIELD[task]) + 1


def averages(tasks: list[str], ours: dict[str, float] | None = None) -> dict[str, float]:
    """Mean score per method over exactly ``tasks``, plus ``"ours"``.

    **Every method is averaged over the same task set**, which is the only way the row is
    internally comparable -- and it is still not the report's Avg AUROC, which covers all
    twelve. Raises rather than skipping when one of ``tasks`` has no score for us, because
    an average silently taken over a smaller set is exactly the flattery this table has
    already been corrected for once.
    """
    ours = OURS if ours is None else ours
    missing = [t for t in tasks if t not in ours]
    if missing:
        raise ValueError(f"no score of ours for {missing}; an average over a different "
                         f"task set than the one claimed is not comparable")
    out = {m: statistics.mean(FIELD[t][m] for t in tasks) for m in METHODS}
    out["ours"] = statistics.mean(ours[t] for t in tasks)
    return out


def average_rank(tasks: list[str], ours: dict[str, float] | None = None) -> tuple[int, int]:
    """Our standing among the methods' *averages* -- a rank of means, not a mean of ranks.

    These differ, and the difference has already been wrong here once: a mean of ranks gave
    a non-integer "6.43" that ranks us against nothing. The row must answer "where does our
    average sit among their averages", which is this.
    """
    avg = averages(tasks, ours)
    beat = sum(1 for m in METHODS if avg[m] > avg["ours"])
    return beat + 1, len(METHODS) + 1


def report(ours: dict[str, float] | None = None) -> str:
    ours = OURS if ours is None else ours
    tasks = [t for t in FIELD if t in ours]
    lines = [f"{'task':<28}{'ours':>8}{'rank':>7}   field best / worst",
             "-" * 72]
    ranks = []
    for t in sorted(tasks, key=lambda t: rank(ours[t], t)[0]):
        r, n = rank(ours[t], t)
        ranks.append(r)
        vals = FIELD[t].values()
        lines.append(f"{t:<28}{ours[t]:>8.2f}{f'{r}/{n}':>7}   "
                     f"{max(vals):.2f} / {min(vals):.2f}")
    avg = averages(tasks, ours)
    ar, an = average_rank(tasks, ours)
    lines += ["-" * 72,
              f"{'average':<28}{avg['ours']:>8.2f}{f'{ar}/{an}':>7}   over {len(tasks)} tasks",
              "",
              f"median rank {int(statistics.median(ranks))} of {len(METHODS) + 1}; "
              f"ranks {', '.join(map(str, sorted(ranks)))}",
              f"tasks not yet run: {', '.join(sorted(set(FIELD) - set(tasks))) or 'none'}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
