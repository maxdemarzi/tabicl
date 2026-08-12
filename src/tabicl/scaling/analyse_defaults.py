"""The phase-2 defaults analysis, written before the data existed.

`RESEARCH.md` §9 fixes the hypothesis, the quantity and the falsifier before any run. This
fixes the ARITHMETIC before any run, which is the other half of the same discipline: an
analysis chosen after seeing the numbers can rescue almost any hypothesis, and the freedom to
choose it is exactly what a pre-registration is supposed to remove. Every decision below --
which arm, which pairing, which correlation, which exclusion, which threshold -- was made with
the logs still being written on a pod.

What it reads
-------------
The `PERSEED_ARM base` line from each run, which is the fixed `base` arm's per-seed test
scores, and the `GAPRATIO` line, which is `(train->test gap) / (training span)`. Runs are named
`<task>_<config>` with config in {ctl, recent1k, cal}, so a directory of logs is a table.

**The `base` arm and not the calibrated summary**, deliberately. A default is a fixed
configuration, and two runs of a fixed configuration correlate at r = 0.88-0.94 across seeds,
so pairing cuts the standard error 2.4-3.0x. Two CALIBRATED runs correlate at 0.34 and -0.03
and cannot be paired at all.

The three questions, in the order they are allowed to be asked
--------------------------------------------------------------
1. **The gate.** `recent` is deterministic, so its eight seeds are eight model fits of ONE
   context and its spread has the draw variance missing entirely. If `recent-half` on
   rel-event -- which samples at random inside the recent half -- collapses toward `ctl`, then
   +6.27 was one lucky draw and questions 2 and 3 are not worth asking. Reported first and
   loudly, because a gate reported after the headline is not a gate.
2. **The pre-registered rule.** Spearman rho between `gap_ratio` and the recency gain across
   the eleven tasks EXCLUDING rel-event, which is in-sample because the rule came from it.
   rho <= -0.5 survives; anything above refutes, including a flat set of near-zero gains,
   which has no rank structure and would mean rel-event-specific with no rule behind it.
3. **The flat question anyway.** Worst-case regret for adopting each candidate as a plain
   default, since that is what "should this ship" means and it is worth having even if the
   rule dies.

Usage
-----
    python -m tabicl.scaling.analyse_defaults run_d2a/results/logs run_d2b/results/logs ...
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

# Fixed here, not chosen later. RESEARCH.md §9.
RHO_THRESHOLD = -0.5
IN_SAMPLE = "rel-event/user-ignore"     # where the rule came from; excluded from its own test
FLOOR = 0.6                             # the project's standing +-0.6 resolution floor

PERSEED = re.compile(r"^PERSEED_ARM\tbase\t(.+)$", re.M)
GAPRATIO = re.compile(r"^GAPRATIO\t(\S+)\t([\d.]+)\t([-\d.]+)\t([\d.]+)$", re.M)
CONFIGS = ("ctl", "recent1k", "cal", "recenthalf")


def read_logs(dirs: list[pathlib.Path]) -> tuple[dict, dict]:
    """(scores, ratios). scores[task][config] = per-seed base-arm array."""
    scores: dict[str, dict[str, np.ndarray]] = {}
    ratios: dict[str, float] = {}
    for d in dirs:
        for f in sorted(d.glob("*.log")):
            stem = f.stem
            cfg = next((c for c in CONFIGS if stem.endswith("_" + c)), None)
            if cfg is None:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            m = PERSEED.search(text)
            g = GAPRATIO.search(text)
            if g:
                ratios[g.group(1)] = float(g.group(4))
            if not m:
                print(f"  {stem}: no PERSEED_ARM base line -- run incomplete, skipped")
                continue
            task = g.group(1) if g else stem[: -len(cfg) - 1]
            scores.setdefault(task, {})[cfg] = np.array(
                [float(x) for x in m.group(1).split()])
    return scores, ratios


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """(mean gain, SE, positive seeds, n) for a - b, paired by seed position."""
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(d.mean()), float(se), int((d > 0).sum()), n


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, written out rather than imported so the tie handling is visible."""
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within ties, which is what makes this Spearman and not something else
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    scores, ratios = read_logs(args.dirs)
    if not scores:
        raise SystemExit("no runs found; expected logs named <task>_{ctl,recent1k,cal}.log")

    # ---- 1. THE GATE, before anything else is reported --------------------------------
    print("=" * 78)
    print("1. THE GATE -- is the recency effect a mechanism or one lucky deterministic draw?")
    print("=" * 78)
    gate = scores.get(IN_SAMPLE, {})
    if "recenthalf" in gate and "ctl" in gate and "recent1k" in gate:
        gh, se_h, pos_h, n_h = paired(gate["recenthalf"], gate["ctl"])
        gr, se_r, _, _ = paired(gate["recent1k"], gate["ctl"])
        print(f"  recent1k  - ctl : {gr:+.2f}  (deterministic: 8 fits of ONE context)")
        print(f"  recenthalf- ctl : {gh:+.2f}  SE {se_h:.2f}  {pos_h}/{n_h}  "
              f"(random inside the recent half -- carries the draw variance)")
        if gh > FLOOR:
            print(f"  GATE PASSED: a randomised recent draw still clears the floor, so the "
                  f"effect survives varying the draw.")
        else:
            print(f"  *** GATE FAILED *** recent-half is inside the +-{FLOOR} floor. The "
                  f"+{gr:.2f} is a property of one draw, not of recency. §9 dies here and "
                  f"the correlation below should not be read as evidence of anything.")
    else:
        print("  recent-half not present yet (lane B) -- gate UNRESOLVED, read on with that "
              "in mind.")

    # ---- 2. THE PRE-REGISTERED RULE ----------------------------------------------------
    print("\n" + "=" * 78)
    print("2. THE PRE-REGISTERED RULE -- rho(gap_ratio, recency gain) <= -0.5, excluding "
          f"{IN_SAMPLE}")
    print("=" * 78)
    print(f"{'task':32s}{'ratio':>8s}{'recency':>9s}{'SE':>6s}{'+/n':>7s}{'calendar':>10s}")
    rows = []
    for task in sorted(scores):
        s = scores[task]
        if "ctl" not in s:
            continue
        rec = paired(s["recent1k"], s["ctl"]) if "recent1k" in s else None
        cal = paired(s["cal"], s["ctl"]) if "cal" in s else None
        r = ratios.get(task, float("nan"))
        rows.append((task, r, rec, cal))
        print(f"{task:32s}{r:8.3f}"
              f"{rec[0] if rec else float('nan'):+9.2f}{rec[1] if rec else float('nan'):6.2f}"
              f"{(str(rec[2]) + '/' + str(rec[3])) if rec else '-':>7s}"
              f"{cal[0] if cal else float('nan'):+10.2f}")

    test = [(r, rec[0]) for t, r, rec, _ in rows
            if rec and t != IN_SAMPLE and not np.isnan(r)]
    if len(test) >= 4:
        rho = spearman(np.array([a for a, _ in test]), np.array([b for _, b in test]))
        print(f"\n  Spearman rho over {len(test)} held-out tasks: {rho:+.3f}   "
              f"(threshold {RHO_THRESHOLD:+.2f}, fixed in RESEARCH.md §9 before any run)")
        if rho <= RHO_THRESHOLD:
            print("  SURVIVES: recency pays where the train->test gap is small relative to "
                  "the training span. It is a rule keyed on a label-free quantity, not a "
                  "default.")
        else:
            print("  REFUTED: the gap ratio does not order where recency pays. If the gains "
                  "are also all near zero, that is the flat outcome the pre-registration "
                  "named -- rel-event-specific, with no rule behind it.")
    else:
        print(f"\n  only {len(test)} held-out tasks so far; the rule needs the full set "
              f"before rho means anything.")

    # ---- 3. THE FLAT QUESTION ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. WORST-CASE REGRET -- should either candidate ship as a plain default?")
    print("=" * 78)
    for name, idx in (("recency", 2), ("calendar", 3)):
        gains = [row[idx][0] for row in rows if row[idx]]
        if not gains:
            continue
        adopt = max(0.0, -min(gains))       # worst loss from turning it on
        decline = max(0.0, max(gains))      # worst forgone gain from leaving it off
        print(f"  {name:9s} adopt {adopt:5.2f}   decline {decline:5.2f}   -> "
              f"{'ADOPT' if adopt < decline else 'DECLINE'}"
              f"   (n={len(gains)}, worst {min(gains):+.2f}, best {max(gains):+.2f})")
    print("\nA regret verdict from a partial set is provisional; both columns move with the "
          "worst and best cell, which is exactly what a missing task might be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
