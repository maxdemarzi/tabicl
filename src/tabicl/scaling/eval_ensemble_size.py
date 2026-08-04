"""Does a feature setting's verdict flip with ensemble size?

Usage
-----
    python -m tabicl.scaling.eval_ensemble_size

UNFINISHED -- this is the open question described in TODO.md. It was killed partway
through its first cells; no result has been produced.

Two runners disagree about the same configuration on rel-event at full context:

    n_estimators=1  ->  plain 80.27, categorical 81.77   (+1.50)
    n_estimators=4  ->  plain 80.77, categorical 77.84   (-2.93)

Row chunking has been cleared as the cause (max|dp| 1.1e-05, AUC identical), and the
seeds are deterministic at full context. `n_estimators` is the only variable left.

If the sign really flips, it is not a curiosity about one feature block. The calibration
committed today selects configurations at --select-estimators 1 to keep the sweep
affordable and then fits the final model at n_estimators=4. A setting whose sign depends
on ensemble size makes that selection unsound, and would explain rel-event's calibrated
78.11 without appealing to noise or distribution shift.

Everything is held fixed except n_estimators: same features, same fit set (train+val),
same random_state, same official test split.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from tabicl import TabICLClassifier
from tabicl.scaling import Table, asof_statistics

DS, TK = "rel-event", "user-ignore"
W = [pd.Timedelta(days=30), pd.Timedelta(days=365)]
DEVICE = os.environ.get('TABICL_DEVICE', 'cpu')
SIZES = tuple(int(x) for x in os.environ.get('ENSEMBLE_SIZES', '1,2,4').split(','))
# 8 was dropped from the default: at full context it reached ~22 GB and drove the
# machine to 854 MB available. 1 vs 4 is the comparison that matters; 2 shows the
# transition. Override with ENSEMBLE_SIZES if there is headroom.
SPECS = [((2, None, False), "plain"), ((2, 4, True), "categorical")]


def numeric(df):
    o = df.copy()
    for c in o.columns:
        if not pd.api.types.is_numeric_dtype(o[c]):
            o[c] = pd.factorize(o[c])[0]
    return np.nan_to_num(o.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


db = get_dataset(DS, download=True).get_db()
task = get_task(DS, TK, download=True)
key, target, ent = task.entity_col, task.target_col, task.entity_table
tr = pd.concat([task.get_table("train", mask_input_cols=False).df,
                task.get_table("val", mask_input_cols=False).df], ignore_index=True)
te = task.get_table("test", mask_input_cols=False).df
tcol = next(c for c in tr.columns if pd.api.types.is_datetime64_any_dtype(tr[c]))
ent_df, pk = db.table_dict[ent].df, db.table_dict[ent].pkey_col
kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
        for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
        if pt == ent and t.time_col][:3]
print(f"{DS}/{TK}  fit={len(tr)}  test={len(te)}  device={DEVICE}  sizes={SIZES}", flush=True)


def build(frame, spec):
    mc, tk, md = spec
    base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
    drop = {target, key, pk, tcol}
    cols = [c for c in base.columns
            if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
    blocks = [base[cols].reset_index(drop=True)]
    for n, fk, tc in kids:
        t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=W,
                  max_columns=mc, top_k_categories=tk, include_mode=md)
        blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                      frame[tcol].to_numpy()).add_prefix(f"{n}__"))
    return pd.concat(blocks, axis=1)


scores = {}
y, y_te = tr[target].to_numpy(), te[target].to_numpy()
for spec, label in SPECS:
    f_tr = build(tr, spec)
    f_te = build(te, spec).reindex(columns=f_tr.columns, fill_value=np.nan)
    X, Xe = numeric(f_tr), numeric(f_te)
    print(f"\n{label}: {X.shape[1]} features", flush=True)
    for n_est in SIZES:
        t0 = time.perf_counter()
        # AMP is on by default and costs 7.3 AUC on this task, so it is disabled for any
        # accuracy comparison. With it off, CUDA reproduces CPU exactly.
        cfg = None if DEVICE == "cpu" else {k: {"use_amp": False}
                                            for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
        kwargs = {"inference_config": cfg} if cfg else {}
        clf = TabICLClassifier(n_estimators=n_est, device=DEVICE, random_state=0,
                               **kwargs).fit(X, y)
        auc = roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100
        scores[(label, n_est)] = auc
        print(f"  n_estimators={n_est:<2}  AUC={auc:.2f}   ({time.perf_counter()-t0:.0f}s)",
              flush=True)

print("\n--- does the verdict depend on ensemble size? ---", flush=True)
print(f"{'n_estimators':>13}  {'plain':>8}  {'categorical':>12}  {'gap':>8}", flush=True)
gaps = []
for n_est in SIZES:
    a, b = scores[("plain", n_est)], scores[("categorical", n_est)]
    gaps.append(b - a)
    print(f"{n_est:>13}  {a:>8.2f}  {b:>12.2f}  {b - a:>+8.2f}", flush=True)

signs = {np.sign(g) for g in gaps}
if len(signs) > 1:
    print("\nVERDICT: the sign FLIPS with ensemble size. Selecting at n_estimators=1 and "
          "scoring at 4 -- which the calibration does -- is unsound.", flush=True)
else:
    print("\nVERDICT: sign is stable across ensemble sizes; n_estimators is not the "
          "explanation and the two runners differ some other way.", flush=True)
