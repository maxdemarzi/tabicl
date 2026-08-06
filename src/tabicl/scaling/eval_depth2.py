"""Does depth-2 help, now that the cutoff actually reaches the grandchildren?

The earlier depth-2 result on rel-trial — 49.12 depth-1 against 49.37 depth-2, both at
chance — was measured while `_aggregate_by_row` passed `None` as the nested cutoff, so every
grandchild row contributed regardless of the entity's prediction time. Those features were
leaky, which makes the number uninformative rather than merely disappointing: a leak usually
*inflates*, and this one produced nothing, so the honest position is that depth-2 has never
been measured cleanly at all.

This uses `flatten_relational`, which already has depth-2 and now propagates the deadline.
No new architecture: rel-trial has the unique entity keys the join path requires, so the
measurement is available today and the lift design can wait on its result.

Paired by seed, one variable: the same child tables with and without their grandchildren.

Usage
-----
    python -m tabicl.scaling.eval_depth2 rel-trial study-outcome
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import Table, flatten_relational

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}


def _numeric(df: pd.DataFrame, cats: dict, fit: bool) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col].astype(object)
        if fit:
            codes, uniques = pd.factorize(s)
            cats[col] = {v: i for i, v in enumerate(uniques)}
            out[col] = codes
        else:
            out[col] = s.map(cats.get(col, {})).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    args = ap.parse_args()

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    y, y_te = train[target].to_numpy(), test[target].to_numpy()

    tables = db.table_dict
    all_kids = [(n, fk) for n, t in tables.items()
                for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                if pt == entity and t.time_col]
    # 0 means all, matching eval_track_record. Slicing [:0] instead gave an empty list and
    # the run reported "0 children, 0 grandchildren" as though the schema had none.
    kids = all_kids if args.children <= 0 else all_kids[: args.children]

    # Prefer children that actually HAVE grandchildren. Taking the first N in dictionary
    # order found none on rel-trial (the subtree is outcomes -> outcome_analyses, and
    # outcomes is not among the first three), while taking all ten OOM-killed the host:
    # the join path is |child| x rows-sharing-a-key, and ten subtrees exceeded memory
    # before the MAX_JOIN_PAIRS guard could fire. Depth-2 is only *about* the children with
    # descendants, so order by that and let --children bound the cost.
    def _has_gc(name):
        ct = tables[name]
        if ct.pkey_col is None:
            return False
        return any(gpt == name and gt.time_col
                   for gt in tables.values()
                   for gpt in (gt.fkey_col_to_pkey_table or {}).values())

    ranked = sorted(all_kids, key=lambda k: (not _has_gc(k[0]), all_kids.index(k)))
    kids = ranked if args.children <= 0 else ranked[: args.children]
    print(f"children chosen (with-grandchildren first): {[k[0] for k in kids]}", flush=True)

    def grandchildren_of(child_name):
        """Tables pointing at this child, which is what depth-2 would add."""
        ct = tables[child_name]
        if ct.pkey_col is None:
            return []
        out = []
        for gn, gt in tables.items():
            for gfk, gpt in (gt.fkey_col_to_pkey_table or {}).items():
                if gpt == child_name and gt.time_col:
                    out.append(Table(gt.df, gfk, gn, time_column=gt.time_col,
                                     max_columns=2))
        return out

    def build(frame, depth):
        specs = []
        for n, fk in kids:
            ct = tables[n]
            gcs = grandchildren_of(n) if depth == 2 else []
            specs.append(Table(ct.df, fk, n, time_column=ct.time_col, max_columns=2,
                               primary_key=ct.pkey_col if gcs else None,
                               children=gcs or None))
        return flatten_relational(frame, key, specs, cutoff_column=tcol)

    n_gc = sum(len(grandchildren_of(n)) for n, _ in kids)
    print(f"{args.dataset}/{args.task}: {len(kids)} children, {n_gc} grandchildren",
          flush=True)
    if not n_gc:
        print("no timestamped grandchild tables -- depth-2 is not available here", flush=True)
        return

    frames = {}
    for depth in (1, 2):
        t0 = time.perf_counter()
        f_tr = build(train, depth)
        f_te = build(test, depth).reindex(columns=f_tr.columns, fill_value=np.nan)
        cats: dict = {}
        frames[depth] = (_numeric(f_tr, cats, True), _numeric(f_te, cats, False))
        print(f"depth {depth}: {f_tr.shape[1]} features in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    def score(X, Xe, rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    print(f"\n{'seed':>5} {'depth1':>9} {'depth2':>9} {'gap':>8}", flush=True)
    gaps = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        n = len(frames[1][0])
        rows = rng.choice(n, size=min(args.context, n), replace=False)
        a = score(*frames[1], rows, seed)
        b = score(*frames[2], rows, seed)
        gaps.append(b - a)
        print(f"{seed:>5} {a:>9.2f} {b:>9.2f} {b - a:>+8.2f}", flush=True)

    g = np.array(gaps)
    print(f"\ndepth2 over depth1: mean {g.mean():+.2f} sd "
          f"{g.std(ddof=1) if len(g) > 1 else 0:.2f} over {len(g)} seeds, "
          f"{(g > 0).sum()}/{len(g)} positive", flush=True)
    print("NOTE: ±0.6 floor. The prior depth-2 figure (49.12 vs 49.37) was measured with "
          "the nested cutoff unpropagated, so those features were leaky and that "
          "comparison is void.", flush=True)


if __name__ == "__main__":
    main()
