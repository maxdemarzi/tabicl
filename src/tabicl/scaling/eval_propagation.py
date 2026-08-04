"""Do neighbours' labels, as features, beat the same pipeline without them?

`RESEARCH.md` item 6b. The feature is already known to be strong in isolation -- neighbour
positive-rate alone scores ROC-AUC ~74 on rel-event with no model involved. What is not
known is whether it adds anything *on top of* the relational features already built, which
is the only question that moves a headline number.

Order of operations, and it is not negotiable: the negative controls run **before** the
comparison. This is the family where a mistake returns a large confident number instead of
an exception, so a lift measured before the controls pass is not evidence of anything.

  1. **Permutation.** Rebuild the propagation features from shuffled training labels and
     score them against the real test labels. They must collapse to chance.
  2. **Temporal.** Move every query cutoff earlier. Withholding history cannot add
     information, so the score must not improve.
  3. **Paired A/B.** Base features against base + propagation, same seeds, same rows.

Usage
-----
    python -m tabicl.scaling.eval_propagation [--seeds 5] [--no-horizon]
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
from tabicl.scaling import (
    Table,
    asof_statistics,
    label_homophily,
    neighbour_label_features,
)
from tabicl.scaling._leakage import permutation_control, temporal_control

WINDOWS = [pd.Timedelta(days=30), pd.Timedelta(days=365)]
NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
PROP_COLS = ["nbr__labelled_degree", "nbr__positive_count", "nbr__positive_rate"]


def _numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--no-horizon", action="store_true",
                    help="ignore the label resolution window. Wrong on a temporal split, "
                         "and kept only to measure how much the horizon is worth.")
    args = ap.parse_args()

    db = get_dataset("rel-event", download=True).get_db()
    task = get_task("rel-event", "user-ignore", download=True)
    key, target = task.entity_col, task.target_col
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    horizon = None if args.no_horizon else getattr(task, "timedelta", None)
    print(f"label horizon: {horizon}", flush=True)

    friends = db.table_dict["user_friends"].df[["user", "friend"]].dropna()
    universe = pd.Index(pd.unique(np.concatenate([
        train[key].to_numpy(), test[key].to_numpy(),
        friends["user"].to_numpy(), friends["friend"].to_numpy()])))
    code = {v: i for i, v in enumerate(universe)}
    edges = np.column_stack([friends["user"].map(code).to_numpy(),
                             friends["friend"].map(code).to_numpy()]).astype(np.int64)
    train_nodes = train[key].map(code).to_numpy()
    test_nodes = test[key].map(code).to_numpy()
    y, y_te = train[target].to_numpy(), test[target].to_numpy()

    labels = np.full(len(universe), -1, dtype=np.int64)
    labels[train_nodes] = y
    known = np.isin(edges[:, 0], train_nodes) & np.isin(edges[:, 1], train_nodes)
    stats = label_homophily(edges[known], labels)
    print(f"homophily lift {stats['lift']:+.4f} over {stats['n_edges']:,} edges", flush=True)
    if not np.isfinite(stats["lift"]) or stats["lift"] <= 0.005:
        print("GATE FAILED -- neighbours carry no label information", flush=True)
        return

    # --- base features, identical in both arms ------------------------------------------
    ent_df = db.table_dict[task.entity_table].df
    pk = db.table_dict[task.entity_table].pkey_col
    kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
            for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
            if pt == task.entity_table and t.time_col][:3]

    def build_base(frame):
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

    b_tr = build_base(train)
    b_te = build_base(test).reindex(columns=b_tr.columns, fill_value=np.nan)

    def propagation(query_nodes, query_times, train_labels, shift=None):
        """Propagation block for one query set. Only training labels are ever passed in."""
        times = np.asarray(query_times)
        if shift:
            times = times - pd.Timedelta(days=shift)
        return neighbour_label_features(
            edges,
            label_nodes=train_nodes, label_values=train_labels,
            query_nodes=query_nodes,
            label_times=train[tcol].to_numpy(), query_times=times,
            label_horizon=horizon, n_nodes=len(universe),
        )

    p_te = propagation(test_nodes, test[tcol].to_numpy(), y)
    coverage = (p_te["nbr__labelled_degree"] > 0).mean()
    print(f"test coverage: {coverage:.3f} of queries have a usable labelled neighbour",
          flush=True)

    # --- control 1: permutation ----------------------------------------------------------
    # Built from shuffled training labels and scored against the *real* test labels, using
    # the propagation columns alone -- mixing in the base features would floor the control
    # at the base score and it could never collapse to chance.
    print("\ncontrol 1: permutation", flush=True)

    def prop_only_score(train_labels):
        block = propagation(test_nodes, test[tcol].to_numpy(), train_labels)
        rate = block["nbr__positive_rate"].to_numpy()
        rate = np.where(np.isnan(rate), np.nanmean(train_labels), rate)
        return roc_auc_score(y_te, rate)

    perm = permutation_control(prop_only_score, y, n_permutations=3, tolerance=0.02)
    print(f"  {perm!r}", flush=True)

    # --- control 2: temporal -------------------------------------------------------------
    print("control 2: temporal", flush=True)
    temporal = temporal_control(
        lambda days: prop_only_score_shifted(propagation, test_nodes, test[tcol].to_numpy(),
                                             y, y_te, days),
        shifts=(0.0, 30.0, 90.0),
    )
    print(f"  {temporal!r}", flush=True)

    if not (perm.passed and temporal.passed):
        print("\nCONTROLS FAILED -- not measuring a lift on features that leak", flush=True)
        return
    print("controls passed\n", flush=True)

    # --- paired A/B ----------------------------------------------------------------------
    p_tr = propagation(train_nodes, train[tcol].to_numpy(), y)
    X_base, Xe_base = _numeric(b_tr), _numeric(b_te)
    X_prop = _numeric(pd.concat([b_tr.reset_index(drop=True),
                                 p_tr.reset_index(drop=True)], axis=1))
    Xe_prop = _numeric(pd.concat([b_te.reset_index(drop=True),
                                  p_te.reset_index(drop=True)], axis=1))
    print(f"{X_base.shape[1]} base features -> {X_prop.shape[1]} with propagation, "
          f"{len(X_base)} train rows, context={args.context}", flush=True)

    def score(X, Xe, rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    print(f"{'seed':>5} {'base':>9} {'+prop':>9} {'gap':>8}", flush=True)
    gaps = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(X_base), size=min(args.context, len(X_base)), replace=False)
        t0 = time.perf_counter()
        a = score(X_base, Xe_base, rows, seed)
        b = score(X_prop, Xe_prop, rows, seed)
        gaps.append(b - a)
        print(f"{seed:>5} {a:>9.2f} {b:>9.2f} {b - a:>+8.2f}   "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    gaps = np.array(gaps)
    print(f"\npropagation gap: mean {gaps.mean():+.2f} sd "
          f"{gaps.std(ddof=1) if len(gaps) > 1 else 0:.2f} over {len(gaps)} seeds, "
          f"{(gaps > 0).sum()}/{len(gaps)} positive", flush=True)
    print("NOTE: the +-0.6 floor applies. user_friends carries no edge timestamps, so the "
          "graph is static; the label horizon makes the *labels* causal but not the edges.",
          flush=True)


def prop_only_score_shifted(propagation, nodes, times, train_labels, y_true, days):
    """Score the propagation block alone with every cutoff moved `days` earlier."""
    block = propagation(nodes, times, train_labels, shift=days)
    rate = block["nbr__positive_rate"].to_numpy()
    rate = np.where(np.isnan(rate), np.nanmean(train_labels), rate)
    return roc_auc_score(y_true, rate)


if __name__ == "__main__":
    main()
