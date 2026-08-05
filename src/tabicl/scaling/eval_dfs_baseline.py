"""Deep Feature Synthesis against our flattening layer, same model underneath.

`RESEARCH.md` 6e. These documents describe the package as "a generic flattening pipeline in
front of a stock TabICL", and that phrase does real work — it is what makes trailing RelGNN
sound acceptable. It has never been measured. DFS (Kanter & Veeramachaneni 2015, implemented
in Featuretools) is the standard automated approach to exactly this problem, so it is the
baseline that phrase implicitly claims parity with.

**The comparison holds the model fixed and swaps only the feature builder**, which is the
only way to separate "our aggregation is better" from "TabICL is better than whatever they
ran". Both arms are scored by the same TabICL, same context, same seeds, same protocol:

  * **DFS** — Featuretools `dfs()` over the same tables, with per-row `cutoff_time` so it
    respects the same temporal boundary we do. Without cutoff times DFS would aggregate the
    future and the comparison would be meaningless in our favour's opposite direction.
  * **ours** — `asof_statistics` over the same child tables.

Two outcomes, both worth having. Ours wins and the engineering is justified. Or DFS wins,
and the honest framing becomes "Featuretools plus label features plus a foundation model" —
still a result, and a far cheaper thing to maintain.

Usage
-----
    python -m tabicl.scaling.eval_dfs_baseline rel-f1 driver-top3
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
from tabicl.scaling import Table, asof_statistics

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
WINDOWS = {"rel-trial": (365, 1095), "rel-avito": (7, 30),
           "rel-event": (30, 365), "rel-f1": (365, 1095)}


def _numeric(df: pd.DataFrame, cats: dict, fit: bool) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        if fit:
            codes, uniques = pd.factorize(out[col])
            cats[col] = {v: i for i, v in enumerate(uniques)}
            out[col] = codes
        else:
            out[col] = out[col].map(cats.get(col, {})).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-f1")
    ap.add_argument("task", nargs="?", default="driver-top3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    args = ap.parse_args()

    import featuretools as ft

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
    print(f"{args.dataset}/{args.task}: {len(kids)} child tables, "
          f"train={len(train)} test={len(test)}", flush=True)

    # --- arm 1: Featuretools DFS, with cutoff times ---------------------------------------
    def ww_safe(df: pd.DataFrame, keep: list) -> pd.DataFrame:
        """Featuretools types every column through woodwork, which rejects the mixed-type
        object columns RelBench frames carry. Keep numerics, datetimes and the join keys;
        cast everything else to string. Dropping is safer than guessing a logical type --
        DFS aggregates numerics, so a mistyped free-text column contributes nothing anyway.
        """
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            s = df[col]
            if col in keep or pd.api.types.is_numeric_dtype(s) \
                    or pd.api.types.is_datetime64_any_dtype(s) \
                    or pd.api.types.is_bool_dtype(s):
                out[col] = s
            else:
                out[col] = s.astype("string")
        return out

    def add(es_, name, df, index, time_index=None):
        """Initialise woodwork explicitly before handing the frame over.

        `add_dataframe` raised WoodworkNotInitError on ingestion, and cleaning dtypes did
        not help: woodwork's accessor is not attaching at all, which is a pandas-version
        incompatibility rather than anything about the data. Calling `.ww.init` directly and
        passing the initialised frame bypasses featuretools' own inference path.
        """
        df = df.copy()
        try:
            df.ww.init(index=index, time_index=time_index)
            return es_.add_dataframe(dataframe_name=name, dataframe=df)
        except Exception:
            return es_.add_dataframe(dataframe_name=name, dataframe=df,
                                     index=index, time_index=time_index)

    es = ft.EntitySet(id=args.dataset)
    parent = ent_df.dropna(subset=[pk]).drop_duplicates(subset=[pk]).reset_index(drop=True)
    es = add(es, entity, ww_safe(parent, [pk]), pk)
    for n, fk, tc in kids:
        child = db.table_dict[n].df.dropna(subset=[fk, tc]).reset_index(drop=True)
        child = ww_safe(child, [fk, tc])
        child["_ft_index"] = np.arange(len(child))
        es = add(es, n, child, "_ft_index", tc)
        es = es.add_relationship(entity, pk, n, fk)
    print(f"entityset built: {entity} + {[k[0] for k in kids]}", flush=True)

    def dfs_features(frame):
        cutoff = pd.DataFrame({"instance_id": frame[key].to_numpy(),
                               "time": frame[tcol].to_numpy()})
        # Tuned rather than minimal, so a DFS loss cannot be dismissed as an unfair
        # baseline. Beyond the five statistics our own layer computes, DFS gets the
        # primitives it is actually known for: num_unique and mode (categorical structure
        # we only added late and off by default), trend and time_since_last (temporal shape
        # our fixed windows cannot express), and skew. Windowed equivalents come free from
        # its cutoff-time machinery.
        fm, _ = ft.dfs(entityset=es, target_dataframe_name=entity,
                       cutoff_time=cutoff, max_depth=args.max_depth,
                       agg_primitives=["count", "sum", "mean", "std", "min", "max",
                                       "num_unique", "mode", "skew", "trend",
                                       "time_since_last", "avg_time_between"],
                       trans_primitives=["month", "year", "weekday"],
                       verbose=False, n_jobs=1)
        fm = fm.reset_index(drop=True)
        return fm.select_dtypes(include=[np.number, "object", "category"])

    t0 = time.perf_counter()
    d_tr = dfs_features(train)
    d_te = dfs_features(test).reindex(columns=d_tr.columns, fill_value=np.nan)
    dfs_secs = time.perf_counter() - t0
    print(f"DFS: {d_tr.shape[1]} features in {dfs_secs:.0f}s", flush=True)

    # --- arm 2: ours ----------------------------------------------------------------------
    def ours(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=windows,
                      max_columns=None)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    t0 = time.perf_counter()
    o_tr = ours(train)
    o_te = ours(test).reindex(columns=o_tr.columns, fill_value=np.nan)
    ours_secs = time.perf_counter() - t0
    print(f"ours: {o_tr.shape[1]} features in {ours_secs:.0f}s", flush=True)

    cats_d, cats_o = {}, {}
    Xd, Xd_te = _numeric(d_tr, cats_d, True), _numeric(d_te, cats_d, False)
    Xo, Xo_te = _numeric(o_tr, cats_o, True), _numeric(o_te, cats_o, False)

    def score(X, Xe, rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    print(f"\n{'seed':>5} {'DFS':>9} {'ours':>9} {'ours-DFS':>10}", flush=True)
    gaps, ds, os_ = [], [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(Xo), size=min(args.context, len(Xo)), replace=False)
        a, b = score(Xd, Xd_te, rows, seed), score(Xo, Xo_te, rows, seed)
        ds.append(a); os_.append(b); gaps.append(b - a)
        print(f"{seed:>5} {a:>9.2f} {b:>9.2f} {b - a:>+10.2f}", flush=True)

    g = np.array(gaps)
    print(f"\nDFS {np.mean(ds):.2f}, ours {np.mean(os_):.2f}", flush=True)
    print(f"ours over DFS: mean {g.mean():+.2f} sd "
          f"{g.std(ddof=1) if len(g) > 1 else 0:.2f} over {len(g)} seeds, "
          f"{(g > 0).sum()}/{len(g)} positive", flush=True)
    print(f"feature build: DFS {dfs_secs:.0f}s for {d_tr.shape[1]} columns, "
          f"ours {ours_secs:.0f}s for {o_tr.shape[1]}", flush=True)
    print("NOTE: ±0.6 floor. Same model, same context, same seeds -- only the feature "
          "builder differs. DFS gets per-row cutoff times so it respects the same temporal "
          "boundary; without them it would aggregate the future.", flush=True)


if __name__ == "__main__":
    main()
