"""Which RelBench tasks can this harness actually run, and what would each one buy?

The registry lists 33 datasets where this project knows seven, and far more tasks per dataset
than the twelve we measure -- rel-stack alone registers six. That was found on 2026-08-11 by
printing the registry before asking it for anything, after weeks of rounds that never listed
it. The obvious next move is to run some, and the obvious next mistake is to pick names off
that list and launch: the registry mixes entity classification with regression, multiclass and
recommendation, and this harness scores binary ROC-AUC and nothing else.

So this scans first. It never calls `get_dataset`, which would pull a whole database per
entry. That is necessary and it was not sufficient: **task tables are not small.** The first
version classified a task AFTER `get_task(download=True)`, so every recommendation and
temporal-graph task was downloaded in full and then discarded for being the wrong shape -- one
of them 5.12 GB, against a 30 GB container disk and 33 datasets still to go. Classification
now happens BEFORE any download: `get_task(download=False)` is enough to read the class, and
only entity tasks are fetched. The `tgb*` families are skipped by default for the same reason,
since they are link-prediction benchmarks on multi-gigabyte temporal graphs and none of them
can be scored by this harness.

For each task it reports what a decision actually needs:

  * **task class**, since `RecommendationTask` is a different scoring problem, not a flag;
  * **target dtype and cardinality**, which is what separates binary from regression and
    multiclass -- `task.task_type` is trusted where present and verified against the column
    either way, because a name is not a measurement;
  * **base rate**, because a task at 0.5% positives needs a different protocol than one at 47%
    and that is worth knowing before the round rather than from its variance;
  * **row counts per split**, which is the cost driver and the reason five of our twelve tasks
    need a memory tier rather than a flag.

**What this does NOT tell you** is whether a task is worth running, and the distinction matters
because the reason to widen the benchmark is specific. The A-versus-B decision currently rests
on rel-trial's +2.17 being the largest tuning gain anywhere: drop that one task and always-tune
stops winning. Every RUNNABLE new task is an independent draw on that claim. A task that is
merely runnable adds a row; a task that could plausibly show a large tuning gain or loss tests
the thing the decision hangs on.

Usage
-----
    python -m tabicl.scaling.scan_tasks                    # every dataset in the registry
    python -m tabicl.scaling.scan_tasks rel-trial rel-avito
    python -m tabicl.scaling.scan_tasks --exclude-known    # skip the twelve we already run
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# The twelve this project already measures, so a scan can show only what is new.
KNOWN = {
    ("rel-f1", "driver-top3"), ("rel-f1", "driver-dnf"),
    ("rel-trial", "study-outcome"),
    ("rel-event", "user-ignore"), ("rel-event", "user-repeat"),
    ("rel-avito", "user-visits"), ("rel-avito", "user-clicks"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"), ("rel-stack", "user-badge"),
    ("rel-amazon", "user-churn"), ("rel-amazon", "item-churn"),
}


def classify(y: pd.Series, declared: str | None) -> tuple[str, str]:
    """(verdict, detail) for a target column, from the values rather than from its name.

    `task_type` is read where the release provides it, and checked against the column
    regardless. The two disagreeing is itself worth seeing -- it would mean the registry and
    the data describe different problems, and a round chosen on the label would measure the
    wrong one.
    """
    y = y.dropna()
    if y.empty:
        return "EMPTY", "no non-null targets"
    n = y.nunique()
    if n == 2:
        vals = sorted(y.unique().tolist())
        rate = float((y == vals[-1]).mean())
        return "BINARY", f"values {vals}, base rate {rate:.4f}"
    if pd.api.types.is_float_dtype(y):
        return "REGRESSION", f"{n:,} distinct, range {y.min():.4g}-{y.max():.4g}"
    if n <= 50:
        return "MULTICLASS", f"{n} classes"
    return "REGRESSION", f"{n:,} distinct integer values"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="default: every dataset in the registry")
    ap.add_argument("--exclude-known", action="store_true",
                    help="skip the twelve tasks already in the standing table")
    ap.add_argument("--include-tgb", action="store_true",
                    help="scan the tgbl-/tgbn-/thgl- families too. Off by default: they are "
                         "link-prediction benchmarks on multi-gigabyte temporal graphs, "
                         "scored by MRR rather than ROC-AUC, so this harness cannot run one "
                         "and fetching them is pure download cost.")
    args = ap.parse_args()

    from relbench.datasets import get_dataset_names
    from relbench.tasks import get_task, get_task_names

    names = args.datasets or sorted(get_dataset_names())
    if not args.include_tgb:
        skipped = [n for n in names if n.startswith(("tgbl", "tgbn", "thgl"))]
        names = [n for n in names if n not in skipped]
        if skipped:
            print(f"skipping {len(skipped)} temporal-graph datasets (MRR link prediction, not "
                  f"runnable here): {', '.join(skipped)}", flush=True)
    print(f"scanning {len(names)} dataset(s); task tables only, no databases\n", flush=True)
    rows, failures = [], []
    for ds in names:
        try:
            tasks = sorted(get_task_names(ds))
        except Exception as exc:
            failures.append((ds, "*", f"get_task_names: {exc}"))
            continue
        for tn in tasks:
            if args.exclude_known and (ds, tn) in KNOWN:
                continue
            try:
                # CLASSIFY BEFORE DOWNLOADING. `download=True` here fetched the task tables of
                # every recommendation and temporal-graph task in the registry only to discard
                # them one line later -- 5.12 GB for a single entry, against a 30 GB container
                # disk. The class is available from the object without any table.
                t = get_task(ds, tn, download=False)
                cls = type(t).__name__
                if "Recommendation" in cls or "Link" in cls:
                    rows.append((ds, tn, cls, "RECOMMENDATION", "ranked, not ROC-AUC",
                                 "", "", ""))
                    continue
                t = get_task(ds, tn, download=True)
                tr = t.get_table("train", mask_input_cols=False).df
                va = t.get_table("val", mask_input_cols=False).df
                te = t.get_table("test", mask_input_cols=True).df
                verdict, detail = classify(tr[t.target_col], getattr(t, "task_type", None))
                rows.append((ds, tn, cls, verdict, detail,
                             f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}"))
            except Exception as exc:
                failures.append((ds, tn, f"{type(exc).__name__}: {str(exc)[:90]}"))
            print(".", end="", flush=True)
    print("\n")

    hdr = ("dataset", "task", "class", "verdict", "detail", "train", "val", "test")
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr)))
    print(line(hdr)); print("  ".join("-" * x for x in w))
    for r in sorted(rows, key=lambda r: (r[3] != "BINARY", r[0], r[1])):
        print(line(r))

    runnable = [r for r in rows if r[3] == "BINARY"]
    print(f"\n{len(runnable)} of {len(rows)} scanned tasks are BINARY and runnable as-is; "
          f"{len(rows) - len(runnable)} are not (regression, multiclass or recommendation).")
    # THE POINT OF WIDENING, restated where the list is read. The A/B verdict rests on
    # rel-trial's +2.17 being the largest tuning gain anywhere; every runnable task is an
    # independent draw on that, and a task with a LARGE gain either way is worth more than
    # three that sit inside the floor.
    print("Each runnable task is an independent draw on the claim the A/B decision rests on: "
          "that rel-trial's +2.17 is the largest tuning gain on the benchmark.")
    if failures:
        print(f"\n{len(failures)} could not be scanned:")
        for ds, tn, why in failures:
            print(f"  {ds}/{tn}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
