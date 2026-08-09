"""Is the calibrated grid resolving real differences, or sampling validation noise?

Answered from run logs already on disk: no GPU, no refits, and **no test scores used to
derive anything**. Every quantity below is a validation quantity, so a rule justified by
them is one the calibrated protocol is allowed to adopt. Test appears in the output only as
an observed consequence, in its own column, and is never selected on.

The calibrated protocol takes the argmax of ~9 candidates on validation. Two numbers decide
whether that pick carries information, and the logs already contain both:

    MARGIN   val(winner) - val(runner-up), within one seed. What selection thinks it gained.
    NOISE    sd across seeds of ONE FIXED candidate's val score. What a val score is worth.

Measured over 41 blocks on 2026-08-08: **margin 0.35, noise 0.69, margin < noise in 32 of
34**. The grid is choosing between candidates it cannot tell apart.

The grid is a product of two axes and they do not behave alike. Arm stability averages 83%,
context stability 71%; the same arm won every seed in 21 of 41 blocks, and in 16 of those the
context still wandered, carrying mean sd(test) 0.89 -- above this benchmark's +-0.6 floor.
That is the actionable half: where the arm has settled, the remaining freedom is a coin flip
over context size that moves test by more than we can resolve.

Usage:

    python -m tabicl.scaling.grid_noise <log>...
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

# Two leading spaces exactly: candidate lines are indented under a seed, and the `chosen`
# line is indented differently. Matching loosely here would fold the winner into the grid
# and understate the margin by counting it twice.
CAND = re.compile(r"^\s{2}(\S+)\s+context=(\d+)\s+(?:val|cv)=(\d+\.\d+)")
CHOSEN = re.compile(r"^\s+chosen (\S+) context=(\d+).*?VAL (\d+\.\d+)\s+TEST (\d+\.\d+)")
TOTAL = re.compile(r"^(\S+)/(\S+)\s+CALIBRATED TEST ROC-AUC x100 = (\d+\.\d+)")
# A task HEADER also ends the previous block. Relying on the summary alone is not enough:
# a task killed by a timeout or the OOM killer never prints one, so its seeds run on into
# the next task. That is not hypothetical -- rel-stack/user-engagement timed out after four
# seeds and its 89.33s pooled with rel-amazon/user-churn's 66.94s, giving one 9-seed block
# at "noise" 11.38 and sd(test) 11.80 on tasks whose real sds are 0.20 and 0.15.
HEADER = re.compile(r"^#+ (?:NEW TASK|GRID) :: (\S+?)/(\S+?) ::")


def parse_blocks(text: str) -> list[tuple[str, list, float]]:
    """Split a log into calibrated blocks: (task, [(cands, chosen, val, test)], mean).

    A seed ends at its ``chosen`` line and a block ends at the CALIBRATED TEST summary.
    **Flushing at the block boundary is the point.** The sibling parser in this project
    (`reanalyse_logs`) once reported +17.66 on a task with a two-point range because it
    flushed only at arm headers and let two blocks run together; the number was impossible
    rather than merely wrong, which is the only reason it was caught. Same failure shape
    here, so the same guard -- and `block_stats` refuses to mix candidate sets.
    """
    out: list[tuple[str, list, float]] = []
    cur: list = []
    seed: dict = {}
    for line in text.splitlines():
        m = CAND.match(line)
        if m:
            seed[(m.group(1), int(m.group(2)))] = float(m.group(3))
            continue
        m = CHOSEN.match(line)
        if m:
            if seed:
                cur.append((dict(seed), (m.group(1), int(m.group(2))),
                            float(m.group(3)), float(m.group(4))))
            seed = {}
            continue
        m = TOTAL.match(line)
        if m:
            if cur:
                out.append((f"{m.group(1)}/{m.group(2)}", cur, float(m.group(3))))
            cur, seed = [], {}
            continue
        if HEADER.match(line):
            # Discard, rather than emit: without a summary there is no task-level score to
            # attach, and a block whose owner died is exactly the one not to report.
            cur, seed = [], {}
    return out


def block_stats(seeds: list) -> dict | None:
    """Margin, noise and stability for one block, or None if it cannot support them.

    Returns None below three seeds: the sd of two numbers is not a noise estimate, and
    reporting one would put a confident ratio on a denominator that is half a coin flip.
    The candidate set is intersected across seeds so a grid that changed mid-block (a task
    whose cap moved, for instance) contributes only the candidates every seed actually
    scored -- comparing a winner against a runner-up that one seed never measured is how a
    margin gets invented.
    """
    if len(seeds) < 3:
        return None
    grid = set.intersection(*(set(s[0]) for s in seeds))
    if len(grid) < 2:
        return None
    noise = statistics.mean(statistics.stdev([s[0][k] for s in seeds]) for k in grid)
    margins, tests, winners = [], [], []
    for cand, chosen, _v, test in seeds:
        ordered = sorted(cand[k] for k in grid)
        margins.append(ordered[-1] - ordered[-2])
        tests.append(test)
        winners.append(chosen)
    margin = statistics.mean(margins)
    arms = [w[0] for w in winners]
    ctxs = [w[1] for w in winners]
    return {
        "k": len(grid), "seeds": len(seeds), "margin": margin, "noise": noise,
        "ratio": margin / noise if noise else float("inf"),
        "stability": max(winners.count(w) for w in set(winners)) / len(winners),
        "arm_stability": max(arms.count(a) for a in set(arms)) / len(arms),
        "ctx_stability": max(ctxs.count(c) for c in set(ctxs)) / len(ctxs),
        "n_arms": len(set(arms)), "n_ctx": len(set(ctxs)),
        "sd_test": statistics.stdev(tests),
    }


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(p) for a in args for p in Path().glob(a)] or sorted(Path().glob("*.log"))
    rows = []
    for path in paths:
        for task, seeds, _mean in parse_blocks(path.read_text(errors="ignore")):
            st = block_stats(seeds)
            if st:
                rows.append((task, path.name, st))
    if not rows:
        # Not a null result -- a measurement that did not happen. Saying so is the whole
        # lesson of the coverage probe that printed "NO PATHOLOGY ANYWHERE ELSE" having
        # measured 1 of 13 tasks.
        print("NO CALIBRATED BLOCKS PARSED -- nothing was measured, which is not a finding.")
        return
    print(f"{'task':<26}{'K':>3}{'sd':>4}{'margin':>8}{'noise':>7}{'m/n':>6}"
          f"{'arm':>6}{'ctx':>6}{'sd(test)':>10}  log")
    for task, log, s in sorted(rows, key=lambda r: r[2]["ratio"]):
        print(f"{task:<26}{s['k']:>3}{s['seeds']:>4}{s['margin']:>8.2f}{s['noise']:>7.2f}"
              f"{s['ratio']:>6.2f}{s['arm_stability']:>6.0%}{s['ctx_stability']:>6.0%}"
              f"{s['sd_test']:>10.2f}  {log}")
    stats = [r[2] for r in rows]
    coin = [s for s in stats if s["ratio"] < 1.0]
    settled = [s for s in stats if s["n_arms"] == 1]
    wander = [s for s in settled if s["n_ctx"] > 1]
    print(f"\n{len(rows)} blocks. Pooled margin {statistics.mean(s['margin'] for s in stats):.2f}, "
          f"pooled noise {statistics.mean(s['noise'] for s in stats):.2f}.")
    print(f"Margin under one seed-sd of val noise: {len(coin)}/{len(stats)} blocks.")
    print(f"Mean arm stability {statistics.mean(s['arm_stability'] for s in stats):.0%}, "
          f"context {statistics.mean(s['ctx_stability'] for s in stats):.0%}.")
    print(f"Same arm every seed: {len(settled)}/{len(stats)}; context still wandered in "
          f"{len(wander)}"
          + (f", mean sd(test) {statistics.mean(s['sd_test'] for s in wander):.2f}"
             if wander else ""))


if __name__ == "__main__":
    main()
