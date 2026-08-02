"""Do the relational features actually help TabICL, on a real task?

RelBench rel-f1 / driver-top3: predict whether a driver finishes top-3, from a
timestamped prediction table over a genuine multi-table schema. The entity table
carries almost no signal on its own, so any lift has to come from the history that
`flatten_relational` aggregates -- under a per-row cutoff, so nothing leaks.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import Table, flatten_relational

TARGET = "qualifying"


def build(entity_df, db, with_history: bool) -> pd.DataFrame:
    """Entity attributes only, or entity attributes plus aggregated history."""
    drivers = db.table_dict["drivers"].df
    base = entity_df.merge(drivers, on="driverId", how="left")

    keep = [c for c in base.columns if c not in {TARGET}]
    base = base[keep]

    if not with_history:
        return base.drop(columns=["driverId", "date"], errors="ignore")

    children = [
        Table(db.table_dict["results"].df, "driverId", "res", time_column="date"),
        Table(db.table_dict["standings"].df, "driverId", "stand", time_column="date"),
        Table(db.table_dict["qualifying"].df, "driverId", "qual", time_column="date"),
    ]
    return flatten_relational(base, "driverId", children, cutoff_column="date")


def numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    db = get_dataset("rel-f1", download=True).get_db()
    task = get_task("rel-f1", "driver-top3", download=True)
    train = task.get_table("train").df
    val = task.get_table("val").df  # test has no labels

    y_tr = train[TARGET].to_numpy()
    y_va = val[TARGET].to_numpy()
    print(f"train={len(train)}  val={len(val)}  positive rate={y_tr.mean():.3f}")

    for label, with_history in (("entity table only", False), ("+ relational history", True)):
        t0 = time.perf_counter()
        f_tr = build(train, db, with_history)
        f_va = build(val, db, with_history)
        f_va = f_va.reindex(columns=f_tr.columns, fill_value=np.nan)
        build_s = time.perf_counter() - t0

        X_tr, X_va = numeric(f_tr), numeric(f_va)

        clf = TabICLClassifier(n_estimators=4, device="cpu", random_state=0)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])

        from sklearn.ensemble import HistGradientBoostingClassifier

        gbdt = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
        auc_gbdt = roc_auc_score(y_va, gbdt.predict_proba(X_va)[:, 1])

        print(
            f"  {label:<22} features={X_tr.shape[1]:>4}  "
            f"TabICL AUC={auc:.4f}   GBDT AUC={auc_gbdt:.4f}   (features built in {build_s:.1f}s)"
        )


if __name__ == "__main__":
    main()
