"""Do graph/motif features add anything on top of relational aggregates?

RelBench rel-event / user-ignore, which has a genuine user-user friendship graph
(`user_friends`, 30.4M edges). Three nested feature sets:

  1. entity table only
  2. + relational history (flatten_relational, per-row cutoff)
  3. + motif features (degree / triangles / clustering over the friendship graph)

Caveat, stated because it matters: `user_friends` carries no timestamp, so the graph
is static. RelBench models it that way, but a friendship formed after a prediction
time is still visible to that prediction, so any lift from (3) is an upper bound on
what a strictly causal version would give.

Usage
-----
    python -m tabicl.scaling.eval_relbench_motif [--device auto] [--n-estimators 8]
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl.scaling import Table, flatten_relational, motif_features, native_available
from tabicl.scaling._evalcfg import add_model_args, describe_model, make_classifier

TASK = "user-ignore"


def numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    args = add_model_args(argparse.ArgumentParser()).parse_args()

    db = get_dataset("rel-event", download=True).get_db()
    task = get_task("rel-event", TASK, download=True)
    key, target = task.entity_col, task.target_col
    train, val = task.get_table("train").df, task.get_table("val").df
    print(f"task={TASK} entity_col={key!r} target={target!r} "
          f"train={len(train)} val={len(val)} pos={train[target].mean():.3f}")
    print(f"compiled WCOJ backend: {native_available()}")
    print(describe_model(args))

    users_tbl = db.table_dict["users"].df
    interest = db.table_dict["event_interest"].df

    # --- motifs over the whole friendship graph ---------------------------------
    # Null foreign keys become -1 under factorize, which the kernels reject.
    friends = db.table_dict["user_friends"].df[["user", "friend"]].dropna()
    t0 = time.perf_counter()
    both = np.concatenate([friends["user"].to_numpy(), friends["friend"].to_numpy()])
    codes, uniques = pd.factorize(both)
    edges = np.column_stack([codes[: len(friends)], codes[len(friends):]]).astype(np.int64)
    print(f"graph: {len(edges):,} edges over {len(uniques):,} users "
          f"(factorized in {time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    motifs = motif_features(edges)
    motifs.index = uniques[motifs.index.to_numpy()]
    elapsed = time.perf_counter() - t1
    print(f"motif features: {len(motifs):,} nodes, {int(motifs['triangles'].sum())//3:,} "
          f"triangles, in {elapsed:.1f}s")

    def build(entity_df, level):
        base = entity_df.merge(users_tbl, left_on=key, right_on="user_id", how="left")
        base = base[[c for c in base.columns if c != target]]
        if level == 1:
            return base.drop(columns=[key, "timestamp", "index", "user_id"], errors="ignore")
        feats = flatten_relational(
            base.drop(columns=["index"], errors="ignore"),
            key,
            [Table(interest, "user", "int", time_column="timestamp")],
            cutoff_column="timestamp",
        )
        if level == 3:
            joined = motifs.reindex(entity_df[key].to_numpy())
            for col in motifs.columns:
                feats[f"friend__{col}"] = joined[col].to_numpy()
        return feats

    y_tr, y_va = train[target].to_numpy(), val[target].to_numpy()
    for level, label in ((1, "entity table only"), (2, "+ relational history"), (3, "+ motif features")):
        f_tr, f_va = build(train, level), build(val, level)
        f_va = f_va.reindex(columns=f_tr.columns, fill_value=np.nan)
        X_tr, X_va = numeric(f_tr), numeric(f_va)

        clf = make_classifier(args).fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        gbdt = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
        auc_g = roc_auc_score(y_va, gbdt.predict_proba(X_va)[:, 1])
        print(f"  {label:<22} features={X_tr.shape[1]:>4}  TabICL AUC={auc:.4f}   GBDT AUC={auc_g:.4f}")

    covered = motifs.reindex(train[key].to_numpy())["degree"].notna().mean()
    print(f"\ntask users present in the friendship graph: {covered:.1%}")


if __name__ == "__main__":
    main()
