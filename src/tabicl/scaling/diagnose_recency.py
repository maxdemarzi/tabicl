"""Why does a recent context help ONE task by +6.27 and no other?

`RESEARCH.md` §9 refuted the temporal-geometry rule and left a hole. Context recency is worth
**+6.27** on rel-event/user-ignore, **−0.31** on rel-event/user-repeat -- same database, same
147-day span, same train→test gap -- and −1.78 to −9.25 on every other task measured. Nothing
computable from the calendar separates them, which is exactly what the pre-registered rule
established by failing.

So the answer, if there is one, is in what the rows ARE rather than when they are. This
computes distributional facts about the context a recency draw actually selects, with **no
model fits at all**, so it costs minutes and can be run over many tasks:

  * **label shift** -- base rate in full train, in the most recent `k` rows, and in test. A
    context whose label distribution matches test is a different object to condition on than
    one that does not, and in-context learning conditions on exactly that.
  * **entity overlap** -- what fraction of test rows have their entity present in the context.
    This project has already measured entity overlap driving a selection failure (val 81.2%
    against test 58.6%), so it is a known-live mechanism here rather than a guess.
  * **window width** -- how many DAYS the most recent `k` rows actually span. rel-event's whole
    training period is 147 days; rel-f1's is 19,860. "The most recent 1,000 rows" is days of
    history on one and years on the other, and that is invisible to any ratio.
  * **entity concentration** -- rows per entity in the draw. A recent window that repeatedly
    samples the same few users is a narrower context than its row count suggests.

**THIS IS EXPLORATORY AND GENERATES HYPOTHESES, NOT RESULTS.** It is run after seeing the
outcome it is meant to explain, on the tasks that produced it, which is the position every
refuted item in `RESEARCH.md` was in before it was refuted. Whatever separates user-ignore here
has to be restated as a prediction and tested on tasks that did not inform it. Anything else is
choosing the explanation after seeing the answer, which is the specific error this project has
made and caught repeatedly.

Usage
-----
    python -m tabicl.scaling.diagnose_recency rel-event user-ignore rel-event user-repeat
    python -m tabicl.scaling.diagnose_recency rel-event user-ignore --k 1000
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def describe(dataset: str, task_name: str, k: int) -> dict:
    from relbench.tasks import get_task

    t = get_task(dataset, task_name, download=True)
    tr = t.get_table("train", mask_input_cols=False).df
    te = t.get_table("test", mask_input_cols=False).df
    tcol, ycol = t.time_col, t.target_col
    ecol = getattr(t, "entity_col", None) or next(
        c for c in tr.columns if c not in (tcol, ycol))

    order = np.argsort(tr[tcol].to_numpy(), kind="stable")
    recent = tr.iloc[order[-k:]]

    t_tr = tr[tcol].to_numpy()
    span = (t_tr.max() - t_tr.min()) / np.timedelta64(1, "D")
    win = (recent[tcol].max() - recent[tcol].min()) / np.timedelta64(1, "D")
    gap = (te[tcol].to_numpy().min() - t_tr.max()) / np.timedelta64(1, "D")

    ent_full, ent_recent = set(tr[ecol]), set(recent[ecol])
    te_ent = te[ecol]

    return {
        "task": f"{dataset}/{task_name}",
        "rows": len(tr),
        "span_d": span,
        "gap_d": gap,
        # THE ABSOLUTE WIDTH, which no ratio carries: the same k rows are days of history on
        # one task and years on another.
        "recent_win_d": win,
        "base_full": float(tr[ycol].mean()),
        "base_recent": float(recent[ycol].mean()),
        "base_test": float(te[ycol].mean()),
        # Does the recent draw move the label distribution TOWARD test, or away from it?
        "shift_full": abs(float(tr[ycol].mean()) - float(te[ycol].mean())),
        "shift_recent": abs(float(recent[ycol].mean()) - float(te[ycol].mean())),
        "ovl_full": float(te_ent.isin(ent_full).mean()),
        "ovl_recent": float(te_ent.isin(ent_recent).mean()),
        "ent_full": len(ent_full),
        "ent_recent": len(ent_recent),
        "rows_per_ent_recent": len(recent) / max(1, len(ent_recent)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="dataset task [dataset task ...]")
    ap.add_argument("--k", type=int, default=1000,
                    help="context size the recency draw takes; 1000 is what §9 measured")
    args = ap.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("pairs must be dataset/task couples")

    rows = []
    for i in range(0, len(args.pairs), 2):
        ds, tn = args.pairs[i], args.pairs[i + 1]
        try:
            rows.append(describe(ds, tn, args.k))
            print(f"  {ds}/{tn} ok", flush=True)
        except Exception as exc:
            print(f"  {ds}/{tn} FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    if not rows:
        raise SystemExit("nothing described")

    print(f"\n{'task':28s}{'rows':>9s}{'span_d':>8s}{'win_d':>8s}{'gap_d':>7s}")
    for r in rows:
        print(f"{r['task']:28s}{r['rows']:9,d}{r['span_d']:8.0f}{r['recent_win_d']:8.2f}"
              f"{r['gap_d']:7.0f}")

    print(f"\n{'task':28s}{'base_full':>10s}{'base_rec':>9s}{'base_test':>10s}"
          f"{'|shift|full':>12s}{'|shift|rec':>11s}")
    for r in rows:
        print(f"{r['task']:28s}{r['base_full']:10.4f}{r['base_recent']:9.4f}"
              f"{r['base_test']:10.4f}{r['shift_full']:12.4f}{r['shift_recent']:11.4f}")

    print(f"\n{'task':28s}{'ovl_full':>9s}{'ovl_rec':>9s}{'ent_full':>10s}{'ent_rec':>9s}"
          f"{'rows/ent':>9s}")
    for r in rows:
        print(f"{r['task']:28s}{r['ovl_full']:9.3f}{r['ovl_recent']:9.3f}"
              f"{r['ent_full']:10,d}{r['ent_recent']:9,d}{r['rows_per_ent_recent']:9.2f}")

    print("\nEXPLORATORY. Run after the outcome it explains, on the tasks that produced it. "
          "Anything that separates the tasks here is a HYPOTHESIS and has to be restated as a "
          "prediction and tested where it did not come from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
