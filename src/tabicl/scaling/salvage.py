"""Recover a result from a run that died before printing its summary.

`eval_track_record --calibrated` prints one ``chosen … VAL x TEST y`` line per seed as that
seed finishes, and a single ``CALIBRATED TEST ROC-AUC`` summary at the end. A run killed by
a timeout or the OOM killer loses only the **summary**; every seed that completed is still
in the log. Discarding those is throwing away hours of measured GPU time.

**What this refuses to do is pretend a partial run is a complete one.** Every result carries
its seed count and a ``complete`` flag saying whether the summary line was present, and
`format_result` prints the count inline so a 3-seed mean can never be read as a 5-seed one.
That distinction is the whole reason this module exists rather than a regex at a call site.

This matters because rel-stack/user-engagement costs ~55 minutes per seed once disk
offloading is engaged, against a 3-hour per-task ceiling -- so the summary line is the part
most likely to be missing, on exactly the tasks that were hardest to measure at all.

    python -m tabicl.scaling.salvage <log>...
"""

from __future__ import annotations

import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

TASK = re.compile(r"^#+ (?:NEW TASK|GRID) :: (\S+?)/(\S+?) ::")
CHOSEN = re.compile(r"^\s+chosen (\S+) context=(\d+).*?VAL (\d+\.\d+)\s+TEST (\d+\.\d+)")
SUMMARY = re.compile(r"^(\S+)/(\S+)\s+CALIBRATED TEST ROC-AUC x100 = (\d+\.\d+)")
EXITED = re.compile(r"^EXIT (\S+?)/(\S+?) rc=(\d+)(.*)$")


@dataclass
class Result:
    task: str
    tests: list[float]
    complete: bool
    exit_note: str = ""

    @property
    def mean(self) -> float:
        return statistics.mean(self.tests)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.tests) if len(self.tests) > 1 else 0.0


def salvage(text: str) -> list[Result]:
    """Every task in the log, with the per-seed test scores that actually completed."""
    out: list[Result] = []
    cur: Result | None = None
    for line in text.splitlines():
        m = TASK.match(line)
        if m:
            if cur is not None:
                out.append(cur)
            cur = Result(f"{m.group(1)}/{m.group(2)}", [], False)
            continue
        if cur is None:
            continue
        m = CHOSEN.match(line)
        if m:
            cur.tests.append(float(m.group(4)))
            continue
        if SUMMARY.match(line):
            cur.complete = True
            continue
        m = EXITED.match(line)
        if m and m.group(3) != "0":
            cur.exit_note = f"rc={m.group(3)}{m.group(4).rstrip()}"
    if cur is not None:
        out.append(cur)
    return [r for r in out if r.tests or r.exit_note]


def format_result(r: Result) -> str:
    """One line, with the seed count always visible.

    A partial mean is a real measurement and a weaker one. It is printed with its ``n`` and
    a PARTIAL marker so it cannot be transcribed into a table as though it were the
    standard five, which is the specific mistake this project has made in other forms.
    """
    if not r.tests:
        return f"{r.task:<28} NO SEEDS COMPLETED -- missing measurement ({r.exit_note})"
    tag = "" if r.complete else "  PARTIAL"
    note = f"  [{r.exit_note}]" if r.exit_note else ""
    return (f"{r.task:<28}{r.mean:>7.2f} +- {r.sd:.2f} over {len(r.tests)} seed"
            f"{'s' if len(r.tests) != 1 else ''}"
            f"{tag}{note}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(p) for a in args for p in Path().glob(a)] or sorted(Path().glob("*.log"))
    found = False
    for path in paths:
        results = salvage(path.read_text(errors="ignore"))
        if not results:
            continue
        found = True
        print(f"== {path.name}")
        for r in results:
            print("  " + format_result(r))
    if not found:
        print("NO TASKS FOUND -- nothing was salvaged, which is not the same as nothing "
              "having run.")


if __name__ == "__main__":
    main()
