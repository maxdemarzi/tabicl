"""Does a context of graph neighbours beat a random context of the same size?

`RESEARCH.md` item 8. rel-event is the task for it: `user_friends` is a genuine
user-user graph (30.4M edges) and `user-ignore` is a per-user label, so a query's
neighbours are other labelled users.

Two stages, and the first can stop the second:

1. **Homophily gate.** If a user's label is uncorrelated with their friends' labels, no
   context construction can help. One number, computed before anything expensive.
2. **Paired comparison at equal context size.** Graph-neighbour context against random,
   same size, same seed, same features. Equal size is what makes it a test of *selection*
   rather than of context length, which is a separate and already-measured lever.

Caveat recorded in `DESIGN.md` and repeated here because it bounds the result:
`user_friends` carries no timestamp, so the graph is static and a friendship formed after
a prediction time is still visible. Any lift measured here is an upper bound on what a
strictly causal version would give.

Usage
-----
    python -m tabicl.scaling.eval_graph_context [--context 5000] [--seeds 3]
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
from tabicl.scaling import Table, asof_statistics, label_homophily, select_graph_context

WINDOWS = [pd.Timedelta(days=30), pd.Timedelta(days=365)]
NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}


def _numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--calibrated", action="store_true",
                    help="full protocol: choose selection method and context size on "
                         "validation, then score test once. This is what makes the "
                         "result eligible for the headline table.")
    ap.add_argument("--hops", type=int, default=1,
                    help="neighbourhood radius; beyond 2 the reachable set saturates "
                         "toward the whole component and selection stops selecting")
    args = ap.parse_args()

    db = get_dataset("rel-event", download=True).get_db()
    task = get_task("rel-event", "user-ignore", download=True)
    key, target = task.entity_col, task.target_col
    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))

    # --- shared id space over users appearing in the task or the graph -----------------
    friends = db.table_dict["user_friends"].df[["user", "friend"]].dropna()
    universe = pd.Index(pd.unique(np.concatenate([
        train[key].to_numpy(), val[key].to_numpy(), test[key].to_numpy(),
        friends["user"].to_numpy(), friends["friend"].to_numpy()])))
    code = {v: i for i, v in enumerate(universe)}
    edges = np.column_stack([friends["user"].map(code).to_numpy(),
                             friends["friend"].map(code).to_numpy()]).astype(np.int64)
    train_nodes = train[key].map(code).to_numpy()
    val_nodes = val[key].map(code).to_numpy()
    test_nodes = test[key].map(code).to_numpy()

    # --- stage 1: the gate --------------------------------------------------------------
    # Only train users have known labels; everyone else is masked to a sentinel class so
    # they cannot inflate the same-label count.
    labels = np.full(len(universe), -1, dtype=np.int64)
    labels[train_nodes] = train[target].to_numpy()
    known = np.isin(edges[:, 0], train_nodes) & np.isin(edges[:, 1], train_nodes)
    stats = label_homophily(edges[known], labels)
    print(f"\nHOMOPHILY on {stats['n_edges']:,} label-known edges: "
          f"observed={stats['observed']:.4f} expected={stats['expected']:.4f} "
          f"lift={stats['lift']:+.4f}", flush=True)
    if not np.isfinite(stats["lift"]) or stats["lift"] <= 0.005:
        print("GATE FAILED: neighbours carry no label information. "
              "Items 6b, 6c and 8 cannot help on this task.", flush=True)
        return
    print("gate passed -- neighbours carry label information\n", flush=True)

    # --- features (identical for both arms) ---------------------------------------------
    ent_df = db.table_dict[task.entity_table].df
    pk = db.table_dict[task.entity_table].pkey_col
    kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
            for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
            if pt == task.entity_table and t.time_col][:3]

    def build(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS, max_columns=2)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    f_tr = build(train)
    f_te = build(test).reindex(columns=f_tr.columns, fill_value=np.nan)
    X, Xe = _numeric(f_tr), _numeric(f_te)
    y, y_te = train[target].to_numpy(), test[target].to_numpy()
    print(f"{X.shape[1]} features, {len(X)} train rows, context={args.context}, hops={args.hops}", flush=True)

    node_to_row = {n: i for i, n in enumerate(train_nodes)}

    def score(rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    if args.calibrated:
        # Selection on validation only; test is touched once, at the end. The candidates
        # are (method, context size), so the choice of *whether* to use the graph is made
        # the same way as every other setting rather than assumed.
        f_va = build(val).reindex(columns=f_tr.columns, fill_value=np.nan)
        Xva, y_va = _numeric(f_va), val[target].to_numpy()

        def fit_eval(rows, Xe_, y_):
            clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                                   random_state=0, inference_config=NOAMP).fit(X[rows], y[rows])
            return roc_auc_score(y_, clf.predict_proba(Xe_)[:, 1]) * 100

        def rows_for(method, size, queries, seed=0):
            if method == "random":
                return np.random.default_rng(seed).choice(len(X), size=min(size, len(X)),
                                                          replace=False)
            picked = select_graph_context(edges, train_nodes, queries, n_context=size,
                                          hops=args.hops, random_state=seed)
            rows = np.array([node_to_row[n] for n in picked if n in node_to_row])
            if len(rows) < min(size, len(X)):
                spare = np.setdiff1d(np.arange(len(X)), rows)
                need = min(size, len(X)) - len(rows)
                rows = np.concatenate([rows, np.random.default_rng(seed).permutation(spare)[:need]])
            return rows

        best = None
        print("selection (validation only):", flush=True)
        for method in ("random", "graph"):
            for size in (2000, 5000, 10000):
                score_v = fit_eval(rows_for(method, size, val_nodes), Xva, y_va)
                print(f"  {method:<7} context={size:<6} val={score_v:.2f}", flush=True)
                if best is None or score_v > best[0]:
                    best = (score_v, method, size)
        _, method, size = best
        print(f"
chosen: {method} context={size}", flush=True)
        auc = fit_eval(rows_for(method, size, test_nodes), Xe, y_te)
        print(f"rel-event/user-ignore  CALIBRATED TEST ROC-AUC x100 = {auc:.2f}"
              f"   (context selection={method}, size={size}, hops={args.hops})", flush=True)
        return

    print(f"{'seed':>5} {'random':>9} {'graph':>9} {'gap':>8}", flush=True)
    gaps = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        rand_rows = rng.choice(len(X), size=min(args.context, len(X)), replace=False)

        picked = select_graph_context(edges, train_nodes, test_nodes,
                                      n_context=args.context, hops=args.hops, random_state=seed)
        graph_rows = np.array([node_to_row[n] for n in picked if n in node_to_row])
        if len(graph_rows) < len(rand_rows):     # keep the two arms the same size
            spare = np.setdiff1d(np.arange(len(X)), graph_rows)
            graph_rows = np.concatenate(
                [graph_rows, rng.permutation(spare)[: len(rand_rows) - len(graph_rows)]])

        t0 = time.perf_counter()
        a, b = score(rand_rows, seed), score(graph_rows, seed)
        gaps.append(b - a)
        print(f"{seed:>5} {a:>9.2f} {b:>9.2f} {b - a:>+8.2f}   "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    gaps = np.array(gaps)
    print(f"\ngraph-context gap: mean {gaps.mean():+.2f} sd "
          f"{gaps.std(ddof=1) if len(gaps) > 1 else 0:.2f} over {len(gaps)} seeds", flush=True)
    print("NOTE: the +-0.6 measurement floor applies; and user_friends has no timestamp, "
          "so this is an upper bound on a causal version.", flush=True)


if __name__ == "__main__":
    main()
