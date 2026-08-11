"""Same features, two backbones: is our gap featurization or model?

Our peer group on RelBench is the flatten-then-tabular-foundation-model set — TabPFN-REL,
RDBLearn+v3, KumoRFMv2 — which is exactly our shape. **Every one of them runs on TabPFN-3
and we run on TabICL.** So every "our features are behind" conclusion in `PERFORMANCE.md`
is confounded with a backbone difference that has never been separated, and the two imply
completely different work:

  gap is the backbone   ->  support TabPFN as a backend. Engineering, not research.
  gap is the features   ->  reverse-engineer RDBLearn's featurization.
  we exceed them        ->  our aggregation was better all along and TabICL masked it.

`flatten_relational` emits a plain table, so the swap is a drop-in and this is the cleanest
paired comparison available: identical features, identical context rows, identical seeds,
one variable changed.

**What this cannot tell you.** A win here is "our featurization plus a stronger backbone",
not "our method improved" — the backbone is someone else's model. And it is test-side, so a
gain is not table-eligible until the calibrated protocol picks it.

Usage
-----
    python -m tabicl.scaling.eval_backbone rel-event user-ignore --seeds 3
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
from tabicl.scaling._guards import assert_no_perfect_feature

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}

DEFAULT_WINDOWS = {
    "rel-trial": "365,1095", "rel-avito": "7,30",
    "rel-event": "30,365", "rel-f1": "365,1095",
}


def _numeric(df, codes, fit):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col].astype(object)
        if fit:
            f, uniq = pd.factorize(s)
            codes[col] = {v: i for i, v in enumerate(uniq)}
            out[col] = f
        else:
            out[col] = s.map(codes.get(col, {})).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-event")
    ap.add_argument("task", nargs="?", default="user-ignore")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--max-columns", default="4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4,
                    help="matched across both backbones so the comparison stays one "
                         "variable. TabPFN's own default is 8, so this may understate it; "
                         "--tabpfn-default adds a second TabPFN arm at its default.")
    ap.add_argument("--drift-tabpfn", action="store_true",
                    help="add a Drift-Resilient TabPFN arm (arXiv 2411.10634): a PFN "
                         "pretrained on evolving structural causal models, so temporal "
                         "non-stationarity is in the prior. Built for exactly this setting "
                         "-- train/val/test in time order with measured drift.")
    ap.add_argument("--tabfm", action="store_true",
                    help="add a TabFM arm (Google Research, 2026-06-30). Zero-shot, frozen, "
                         "in-context -- our paradigm -- and its weights are on Hugging Face "
                         "without the token gate that has blocked the TabPFN arm since this "
                         "file was written. Weights are non-commercial licensed.")
    ap.add_argument("--tabpfn", action="store_true",
                    help="add the TabPFN arm. Off by default because it needs a "
                         "TABPFN_TOKEN, which this project does not have.")
    ap.add_argument("--tabpfn-default", action="store_true",
                    help="also score TabPFN at its shipped defaults, which is what a user "
                         "of that library would actually get.")
    args = ap.parse_args()

    import torch
    if args.device.startswith("cpu") and torch.cuda.is_available():
        raise SystemExit("refusing to run on CPU while CUDA is available")

    TabPFNClassifier = None
    if args.tabpfn:
        from tabpfn import TabPFNClassifier
        import tabpfn
        print(f"tabpfn {getattr(tabpfn, '__version__', '?')}", flush=True)

    # TabFM: Google Research, released 2026-06-30. Zero-shot, frozen weights, in-context --
    # our exact paradigm -- and unlike TabPFN-3 its weights are on Hugging Face without a
    # token gate, which is why this arm can exist at all. The WEIGHTS are under
    # `tabfm-non-commercial-v1.0`; the code is Apache 2.0.
    # Drift-Resilient TabPFN (arXiv 2411.10634, NeurIPS 2024). A PFN pretrained on synthetic
    # datasets generated by EVOLVING structural causal models, so non-stationarity sits in
    # the prior rather than being handled afterwards. Reports ROC AUC 0.786 -> 0.832 against
    # the strongest baselines across 18 datasets, frozen and without tuning.
    #
    # It is here because our failure mode is exactly what it was built for: train, then
    # validation, then test in time order, with drift we have measured -- entity overlap with
    # train falls 81.2% on validation to 58.6% on test.
    DriftClf = None
    if args.drift_tabpfn:
        # The repo documents `pip install .` and nothing else -- no import path, no class
        # name -- and it installs into a `tabpfn/` directory, so it is a FORK that may
        # collide with the real tabpfn package. Guessing an API cost a pod cycle earlier
        # today (the TabFM loader name was right by luck; its device argument was not), so
        # this searches instead of assuming, and says what it found.
        import importlib
        cands = [("tabpfn_extensions.dist_shift", "DistShiftClassifier"),
                 ("tabpfn.dist_shift", "DistShiftClassifier"),
                 ("tabpfn", "DistShiftClassifier"),
                 ("tabpfn", "DriftResilientTabPFNClassifier")]
        for mod, cls in cands:
            try:
                DriftClf = getattr(importlib.import_module(mod), cls)
                print(f"drift-resilient tabpfn: {mod}.{cls}", flush=True)
                break
            except (ImportError, AttributeError):
                continue
        if DriftClf is None:
            import tabpfn
            names = [n for n in dir(tabpfn) if not n.startswith("_")]
            raise SystemExit(f"could not locate the drift classifier. tabpfn exports: {names}")

    TabFMClassifier = tabfm_model = None
    if args.tabfm:
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        # `device` is a KEYWORD on load() and defaults to None, which lands on CPU. The
        # classifier has no device argument at all -- only `keep_cache_on_device` -- so
        # passing it here is the only way to reach the GPU, and omitting it is silent: the
        # run works, produces correct AUCs, and takes roughly thirty times as long. Measured
        # on this pod: 680% CPU across cores, 0% GPU utilisation, 247 minutes of CPU time
        # before it was killed.
        tabfm_model = tabfm_v1_0_0.load(device=args.device)
        # Structural, not intentional. This project's standing rule is never to fit on CPU
        # when CUDA is present, and it has been broken twice by different mechanisms. A rule
        # enforced by remembering is a rule that gets broken; this one now stops the run.
        if torch.cuda.is_available():
            where = {p.device.type for p in tabfm_model.parameters()}                 if hasattr(tabfm_model, "parameters") else {args.device}
            if where and where != {"cuda"}:
                raise SystemExit(
                    f"TabFM loaded onto {where} while CUDA is available. Refusing: a CPU fit "
                    f"here is ~30x slower and silently produces a timing comparison that "
                    f"means nothing.")
        print(f"tabfm 1.0.0 loaded on {args.device}", flush=True)

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    y, y_te = train[target].to_numpy(), test[target].to_numpy()

    max_cols = None if str(args.max_columns).lower() in ("none", "null", "") else int(args.max_columns)
    windows = [pd.Timedelta(days=int(d))
               for d in DEFAULT_WINDOWS.get(args.dataset, "30,365").split(",")]
    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    all_kids = sorted([(n, fk, t.time_col) for n, t in db.table_dict.items()
                       for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                       if pt == entity and t.time_col], key=lambda k: k[0])
    kids = all_kids if args.children <= 0 else all_kids[: args.children]

    def build(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns if c not in drop
                and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=windows,
                      max_columns=max_cols)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    t0 = time.perf_counter()
    f_tr = build(train)
    f_te = build(test).reindex(columns=f_tr.columns, fill_value=np.nan)
    codes: dict = {}
    X, Xe = _numeric(f_tr, codes, True), _numeric(f_te, codes, False)
    assert_no_perfect_feature(X, y, list(f_tr.columns), context="backbone")
    print(f"{args.dataset}/{args.task}: {X.shape[1]} features, {len(X):,} train rows, "
          f"built in {time.perf_counter() - t0:.0f}s", flush=True)

    def tabicl(rows, seed):
        m = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                             random_state=seed, inference_config=NOAMP)
        return m.fit(X[rows], y[rows]).predict_proba(Xe)[:, 1]

    def tabpfn(rows, seed, n_est):
        # ignore_pretraining_limits: our frames run to hundreds of columns and 10k context
        # rows, above the shipped guard. Refusing would measure the guard, not the model.
        m = TabPFNClassifier(n_estimators=n_est, device=args.device, random_state=seed,
                             ignore_pretraining_limits=True)
        return m.fit(X[rows], y[rows]).predict_proba(Xe)[:, 1]

    def tabfm(rows, seed):
        # max_num_rows defaults to 100 CONTEXT ROWS -- two orders of magnitude below the
        # 10,000 this project runs on TabICL, and the model compensates by ensembling over
        # several sampled contexts. Raised to the run's own context so the comparison is
        # like-for-like on data seen; if the model degrades there, that is a fact about the
        # model at our scale and is what we want to learn.
        m = TabFMClassifier(model=tabfm_model, n_estimators=args.n_estimators,
                            max_num_rows=len(rows), max_num_features=X.shape[1])
        return m.fit(X[rows], y[rows]).predict_proba(Xe)[:, 1]

    arms = [("tabicl", lambda r, s: tabicl(r, s))]
    if args.tabpfn:
        arms.append((f"tabpfn@{args.n_estimators}", lambda r, s: tabpfn(r, s, args.n_estimators)))
        if args.tabpfn_default:
            arms.append(("tabpfn@default", lambda r, s: tabpfn(r, s, 8)))
    if args.tabfm:
        arms.append((f"tabfm@{args.n_estimators}", lambda r, s: tabfm(r, s)))

    def drift(rows, seed):
        m = DriftClf(device=args.device, random_state=seed)
        return m.fit(X[rows], y[rows]).predict_proba(Xe)[:, 1]

    if args.drift_tabpfn:
        arms.append(("drift-tabpfn", lambda r, s: drift(r, s)))

    scores: dict = {name: [] for name, _ in arms}
    print(f"\n{'seed':>5}" + "".join(f"{n:>16}" for n, _ in arms), flush=True)
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(X), size=min(args.context, len(X)), replace=False)
        cells = ""
        for name, fn in arms:
            t1 = time.perf_counter()
            try:
                auc = roc_auc_score(y_te, fn(rows, seed)) * 100
            except Exception as exc:                      # noqa: BLE001
                print(f"\n  {name} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                auc = float("nan")
            scores[name].append(auc)
            cells += f"{auc:>10.2f}({time.perf_counter() - t1:>3.0f}s)"
        print(f"{seed:>5}{cells}", flush=True)

    base = np.array(scores["tabicl"])
    print(flush=True)
    for name, _ in arms:
        v = np.array(scores[name])
        line = (f"BACKBONE\t{args.dataset}/{args.task}\t{name}\t{np.nanmean(v):.2f}\t"
                f"{np.nanstd(v, ddof=1) if len(v) > 1 else 0:.2f}")
        if name != "tabicl":
            d = v - base
            se = np.nanstd(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            line += (f"\tvs tabicl {np.nanmean(d):+.2f}\tSE {se:.2f}\t"
                     f"{int(np.nansum(d > 0))}/{len(d)}")
        print(line, flush=True)
    print("\nOne variable: identical features, identical context rows, identical seeds. "
          "Test-side -- a gain is not table-eligible until the calibrated protocol picks "
          "it, and it would be 'our featurization plus a stronger backbone', not a better "
          "method.", flush=True)


if __name__ == "__main__":
    main()
