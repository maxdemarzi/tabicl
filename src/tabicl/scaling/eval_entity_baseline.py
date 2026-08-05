"""Does the relational machinery beat a GBDT on the entity table alone?

Prompted by Rel-LLM (arXiv 2506.05725) Table 1, which reports a **LightGBM using only the
single entity table** scoring **70.09 on rel-trial** — against our 69.36 with the whole
relational pipeline, and our unaugmented base arm at 63.82. If that reproduces, the
flattening is not paying for itself on that task, and no amount of further feature work
should happen before it is understood.

`RESEARCH.md` 6e asks for this baseline. It was already published; we simply had not looked.

Three arms, chosen to separate two explanations that a two-arm comparison confounds:

  * **GBDT, entity table only** — their baseline. No relations at all.
  * **TabICL, entity table only** — same features, our model. Isolates the *model*.
  * **TabICL, full relational features** — our pipeline. Isolates the *features*.

If GBDT-entity beats TabICL-entity, the model is the problem on this task. If TabICL-entity
beats TabICL-full, the relational features are actively hurting. Those are very different
findings and the middle arm is what tells them apart.

Usage
-----
    python -m tabicl.scaling.eval_entity_baseline rel-trial study-outcome
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import Table, asof_statistics

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
WINDOWS = {"rel-trial": (365, 1095), "rel-avito": (7, 30),
           "rel-event": (30, 365), "rel-f1": (365, 1095)}


def _numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _gbdt(X, y, Xe, y_te, seed):
    """LightGBM where available, since that is what the published baseline used."""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=seed, verbose=-1)
        name = "lightgbm"
    except ImportError:
        model = HistGradientBoostingClassifier(random_state=seed)
        name = "histgb"
    model.fit(X, y)
    return roc_auc_score(y_te, model.predict_proba(Xe)[:, 1]) * 100, name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
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

    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    windows = [pd.Timedelta(days=d) for d in WINDOWS.get(args.dataset, (30, 365))]
    kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
            for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
            if pt == entity and t.time_col][:3]

    def entity_only(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        return base[cols].reset_index(drop=True)

    def with_relations(frame):
        blocks = [entity_only(frame)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=windows,
                      max_columns=2)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    e_tr, e_te = entity_only(train), entity_only(test).reindex(
        columns=entity_only(train).columns, fill_value=np.nan)
    r_tr = with_relations(train)
    r_te = with_relations(test).reindex(columns=r_tr.columns, fill_value=np.nan)
    Xe_only, Xe_only_te = _numeric(e_tr), _numeric(e_te)
    Xr, Xr_te = _numeric(r_tr), _numeric(r_te)
    print(f"{args.dataset}/{args.task}  entity-only {Xe_only.shape[1]} features, "
          f"+relations {Xr.shape[1]}, train={len(train)} test={len(test)}", flush=True)

    def tabicl(X, Xe, rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    print(f"\n{'seed':>5} {'GBDT-entity':>12} {'TabICL-entity':>14} {'TabICL-full':>12}",
          flush=True)
    res = {"gbdt": [], "icl_entity": [], "icl_full": []}
    backend = "?"
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(Xr), size=min(args.context, len(Xr)), replace=False)
        g, backend = _gbdt(Xe_only[rows], y[rows], Xe_only_te, y_te, seed)
        a = tabicl(Xe_only, Xe_only_te, rows, seed)
        b = tabicl(Xr, Xr_te, rows, seed)
        res["gbdt"].append(g); res["icl_entity"].append(a); res["icl_full"].append(b)
        print(f"{seed:>5} {g:>12.2f} {a:>14.2f} {b:>12.2f}", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"\nmeans ({backend}): GBDT-entity {m['gbdt']:.2f}  "
          f"TabICL-entity {m['icl_entity']:.2f}  TabICL-full {m['icl_full']:.2f}", flush=True)
    print(f"  relations are worth   {m['icl_full'] - m['icl_entity']:+.2f} (TabICL-full "
          f"minus TabICL-entity)", flush=True)
    print(f"  our model is worth    {m['icl_entity'] - m['gbdt']:+.2f} (TabICL-entity "
          f"minus GBDT-entity, same features)", flush=True)
    print(f"  whole pipeline vs GBDT {m['icl_full'] - m['gbdt']:+.2f}", flush=True)
    print("NOTE: ±0.6 floor. Rel-LLM Table 1 reports LightGBM entity-only at 70.09 on "
          "rel-trial, 53.05 rel-avito, 79.93 rel-event, 73.92 rel-f1.", flush=True)


if __name__ == "__main__":
    main()
