"""Gate three feature blocks that have never run in production, before spending on them.

`eval_track_record` builds every child `Table` with `windows` and `max_columns` and nothing
else. It has never set `top_k_categories`, `include_mode` or `numeric_booleans`, so three
blocks that exist, are unit-tested and are documented as carrying signal the numeric path
cannot have contributed nothing to any number in the standing table:

  categories  per-category proportions over a top-K codebook, all-history and per window.
              Corpus-relative: describes an entity by its mix over globally common values.
  mode        the modal value over each key's prefix. Entity-relative, so unlike the
              histogram it stays meaningful when a column has 247k distinct values.
  booleans    booleans through the numeric path, which turns a boolean history into its
              *rate*. As categories they get one `nunique`, which is 1 or 2.

This is a gate, not a result. Paired by seed, one variable at a time, against the same base
features -- cheap enough to run all four tasks x four variants, where a calibrated sweep is
not. Anything clearing the +-0.6 floor here earns a calibrated run; anything that does not,
does not. Nothing here is eligible for the headline table: no validation split is consulted,
so a gain measured here is a gain *observed on test*, which is exactly the thing this
project keeps having to withdraw.

Usage
-----
    python -m tabicl.scaling.eval_feature_gate rel-trial study-outcome --variant categories
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

# Same per-task windows as eval_track_record: trials run for years, ad impressions for days.
DEFAULT_WINDOWS = {
    "rel-trial": "365,1095",
    "rel-avito": "7,30",
    "rel-event": "30,365",
    "rel-f1": "365,1095",
}

VARIANTS = {
    "categories": dict(top_k_categories=4),
    "mode": dict(include_mode=True),
    "booleans": dict(numeric_booleans=True),
    "all": dict(top_k_categories=4, include_mode=True, numeric_booleans=True),
}


def _numeric(df: pd.DataFrame, codes: dict, fit: bool) -> np.ndarray:
    """Encode to float with one codebook shared by train and test.

    Factorizing per split gives a category code 3 in train and 7 in test, so the model is
    scored on a feature it never saw. That bug has now been found in two runners here.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col].astype(object)
        if fit:
            factorized, uniques = pd.factorize(series)
            codes[col] = {value: i for i, value in enumerate(uniques)}
            out[col] = factorized
        else:
            out[col] = series.map(codes.get(col, {})).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="categories")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--max-columns", default="2")
    ap.add_argument("--category-share", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    args = ap.parse_args()

    # Fitting TabICL on CPU with a GPU present is ~30x slower for an identical answer.
    import torch
    if args.device.startswith("cpu") and torch.cuda.is_available():
        raise SystemExit("refusing to run on CPU while CUDA is available")

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    y, y_te = train[target].to_numpy(), test[target].to_numpy()

    max_cols = None if str(args.max_columns).lower() in ("none", "null", "") else int(args.max_columns)
    spec = DEFAULT_WINDOWS.get(args.dataset, "30,365")
    windows = [pd.Timedelta(days=int(d)) for d in spec.split(",")]

    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    all_kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
                for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                if pt == entity and t.time_col]
    kids = all_kids if args.children <= 0 else all_kids[: args.children]
    print(f"{args.dataset}/{args.task}  variant={args.variant}  windows={spec}  "
          f"max_columns={max_cols}  children={[k[0] for k in kids]}", flush=True)

    def build(frame, extra):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            table = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=windows,
                          max_columns=max_cols, min_category_share=args.category_share,
                          **extra)
            blocks.append(asof_statistics(table, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    frames, widths = {}, {}
    for label, extra in (("base", {}), (args.variant, VARIANTS[args.variant])):
        t0 = time.perf_counter()
        f_tr = build(train, extra)
        f_te = build(test, extra).reindex(columns=f_tr.columns, fill_value=np.nan)
        codes: dict = {}
        frames[label] = (_numeric(f_tr, codes, True), _numeric(f_te, codes, False))
        widths[label] = list(f_tr.columns)
        print(f"{label:>11}: {f_tr.shape[1]} features in {time.perf_counter() - t0:.0f}s",
              flush=True)

    # An empty block does not raise, it reports +0.00 with sd 0.00 -- indistinguishable
    # from a clean null. Depth-2 spent a week "measured at no effect" that way. Two of
    # these variants have their own silent-empty path: `min_category_share` can reject
    # every column on a free-text schema, and a schema with no boolean columns leaves
    # `numeric_booleans` a no-op. Refuse to score identical inputs.
    added = [c for c in widths[args.variant] if c not in set(widths["base"])]
    removed = [c for c in widths["base"] if c not in set(widths[args.variant])]
    if not added and not removed:
        raise SystemExit(
            f"variant {args.variant!r} changed no columns ({len(widths['base'])} either "
            f"way), so any gap measured here would be an artefact of an empty block. "
            f"Either this schema has nothing for it to act on, or the "
            f"--category-share gate ({args.category_share}) rejected every column."
        )
    print(f"{args.variant}: +{len(added)} columns, -{len(removed)}; "
          f"e.g. {added[:3]}", flush=True)

    def score(X, Xe, rows, seed):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    print(f"\n{'seed':>5} {'base':>9} {args.variant:>11} {'gap':>8}", flush=True)
    gaps = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        n = len(frames["base"][0])
        rows = rng.choice(n, size=min(args.context, n), replace=False)
        a = score(*frames["base"], rows, seed)
        b = score(*frames[args.variant], rows, seed)
        gaps.append(b - a)
        print(f"{seed:>5} {a:>9.2f} {b:>11.2f} {b - a:>+8.2f}", flush=True)

    g = np.array(gaps)
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\nGATEROW\t{args.dataset}/{args.task}\t{args.variant}\t{g.mean():+.2f}\t"
          f"{sd:.2f}\t{(g > 0).sum()}/{len(g)}\t{len(added)}", flush=True)
    print(f"{args.variant} over base: mean {g.mean():+.2f} sd {sd:.2f} over {len(g)} seeds, "
          f"{(g > 0).sum()}/{len(g)} positive", flush=True)
    print("Gate only. +-0.6 floor, and this is test-side -- a pass earns a calibrated run, "
          "not a table entry.", flush=True)


if __name__ == "__main__":
    main()
