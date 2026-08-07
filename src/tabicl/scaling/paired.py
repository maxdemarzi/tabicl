"""Read two runs' `PERSEED` lines and report the paired difference.

Every A/B in this project is paired: seed *i* draws the same context rows and the same model
randomness in both arms. The summary sd the runner prints is the **wrong denominator** for
that comparison — using it once turned a 4.7-SE result into an undecided 1.6 and flipped the
calendar verdict twice.

Despite which, every comparison in this session was computed by hand: ssh into the pod, awk
the `chosen ... TEST x` lines out of the log, paste them into a Python snippet. That is how a
pairing over **10 of 12 seeds** nearly got reported as the twelve-seed answer — the
extraction truncated, and the two missing seeds happened to include the one bad draw, which
would have turned +0.39 into +0.63.

    python -m tabicl.scaling.paired without.log with.log

Reads `PERSEED` from each, pairs by position, and prints the difference with the paired
standard error, a t, the sign count, and what the variance did.

**Read the variance line.** A feature can be worth nothing on the mean and still matter: on
rel-f1/driver-dnf the sibling tables moved the mean +0.61 at t=0.81 — indistinguishable from
zero — while cutting the spread from 2.05 to 0.57 and lifting the worst seed from 65.60 to
69.65. And it can go the other way: depth-2 on rel-avito/user-clicks gained +0.39 on the
mean while **doubling** the spread, with its worst seed 1.45 below the baseline's worst. The
mean alone would have called those the same kind of result.
"""

from __future__ import annotations

import sys

import numpy as np


def read_perseed(path: str) -> list[list[float]]:
    """Every `PERSEED` block in a log, in order. One per calibrated arm."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("PERSEED\t"):
                out.append([float(v) for v in line.strip().split("\t")[1:]])
    return out


def report(a: list[float], b: list[float], label: str = "") -> None:
    if len(a) != len(b):
        # Truncating to the shorter is exactly the mistake this tool exists to prevent: the
        # dropped seeds are not a random subset of the comparison.
        raise SystemExit(
            f"unequal replicate counts ({len(a)} vs {len(b)}) — these are not paired runs. "
            f"Pairing the first {min(len(a), len(b))} would silently drop whichever seeds "
            f"came last, and there is no reason those are the average ones."
        )
    x, y = np.asarray(a, float), np.asarray(b, float)
    d = y - x
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"{label}{'  ' if label else ''}{len(d)} seeds paired")
    print(f"  A {x.mean():6.2f} (sd {x.std(ddof=1):.2f}, worst {x.min():.2f})")
    print(f"  B {y.mean():6.2f} (sd {y.std(ddof=1):.2f}, worst {y.min():.2f})")
    print(f"  delta {d.mean():+6.2f}   paired SE {se:.2f}   t {d.mean() / se:+.2f}   "
          f"{int((d > 0).sum())}/{len(d)} positive")
    print(f"  spread {x.std(ddof=1):.2f} -> {y.std(ddof=1):.2f}   "
          f"worst seed {x.min():.2f} -> {y.min():.2f}")
    verdict = ("above the +-0.6 floor" if abs(d.mean()) >= 0.6
               else "INSIDE the +-0.6 floor -- not resolvable, and not table-eligible")
    print(f"  {verdict}")
    print(f"  per-seed: {np.round(d, 2).tolist()}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    runs = [read_perseed(p) for p in sys.argv[1:3]]
    for i, (path, blocks) in enumerate(zip(sys.argv[1:3], runs)):
        if not blocks:
            raise SystemExit(f"no PERSEED lines in {path}. Runs made before this line "
                             f"existed have to be re-run rather than scraped.")
    for i, (a, b) in enumerate(zip(*runs)):
        report(a, b, label=f"[block {i}]")
        print()


if __name__ == "__main__":
    main()
