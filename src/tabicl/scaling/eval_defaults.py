"""What should the package defaults be? Chosen on validation, across every task.

A default is a choice made *without* seeing the user's data, so it should be the setting
that is robust across tasks — not the one that wins on any particular task. That makes it a
different question from the calibrated table, which picks per task, and it needs its own
measurement: mean validation performance across the whole benchmark, with test untouched.

The defaults currently disagree with themselves, which is what prompted this:

  setting        library `Table`        `eval_track_record`     `STATUS.md` says
  max_columns    None                   2                       "a rescue, not a default"
  windows        () — none at all       per-dataset             "recency is usually the
                                                                 strongest signal a history
                                                                 carries"

So a user calling `flatten_relational` directly gets **no windows and no column budget**,
while every number this project has published was measured *with* windows and a budget of
2. Whatever the right answer is, those three should agree.

One fit per seed per candidate, scored on **validation**. No test data is read at any
point, which is what makes the resulting choice a legitimate default rather than a tuned
one.

Usage
-----
    python -m tabicl.scaling.eval_defaults --axis max-columns
    python -m tabicl.scaling.eval_defaults --axis windows --seeds 3
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

TASKS = [("rel-f1", "driver-top3"), ("rel-trial", "study-outcome"),
         ("rel-event", "user-ignore"), ("rel-avito", "user-visits")]

# Task-scale windows, as the runner uses. The `windows` axis tests these against none at
# all, which is what the library ships.
DEFAULT_WINDOWS = {
    "rel-trial": [365, 1095], "rel-avito": [7, 30],
    "rel-event": [30, 365], "rel-f1": [365, 1095],
}

# Candidate values per axis. Each is a plausible default someone could ship.
AXES = {
    "max-columns": [None, 2, 4],
    "windows": ["task", "none", "single"],
    "children": [3, 6, 0],
}


def _numeric(df, codes, fit):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col].astype(object)
        if fit:
            factorized, uniques = pd.factorize(s)
            codes[col] = {v: i for i, v in enumerate(uniques)}
            out[col] = factorized
        else:
            out[col] = s.map(codes.get(col, {})).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=sorted(AXES), default="max-columns")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    args = ap.parse_args()

    import torch
    if args.device.startswith("cpu") and torch.cuda.is_available():
        raise SystemExit("refusing to run on CPU while CUDA is available")

    values = AXES[args.axis]
    print(f"axis={args.axis} candidates={values} seeds={args.seeds}\n", flush=True)
    table: dict = {}

    for ds, tk in TASKS:
        db = get_dataset(ds, download=True).get_db()
        task = get_task(ds, tk, download=True)
        key, target, entity = task.entity_col, task.target_col, task.entity_table
        train = task.get_table("train", mask_input_cols=False).df
        val = task.get_table("val", mask_input_cols=False).df
        tcol = next(c for c in train.columns
                    if pd.api.types.is_datetime64_any_dtype(train[c]))
        y, y_va = train[target].to_numpy(), val[target].to_numpy()
        ent_df = db.table_dict[entity].df
        pk = db.table_dict[entity].pkey_col
        # Sorted by name, not dict order: dict order picks different tables on different
        # machines, which is not something a default may depend on.
        all_kids = sorted(
            [(n, fk, t.time_col) for n, t in db.table_dict.items()
             for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
             if pt == entity and t.time_col],
            key=lambda k: k[0])

        for value in values:
            max_cols = value if args.axis == "max-columns" else 2
            n_kids = value if args.axis == "children" else 3
            if args.axis == "windows":
                spec = {"task": DEFAULT_WINDOWS[ds], "none": [], "single": [30]}[value]
            else:
                spec = DEFAULT_WINDOWS[ds]
            windows = [pd.Timedelta(days=d) for d in spec]
            kids = all_kids if n_kids <= 0 else all_kids[:n_kids]

            def build(frame):
                base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
                drop = {target, key, pk, tcol}
                cols = [c for c in base.columns if c not in drop
                        and not pd.api.types.is_datetime64_any_dtype(base[c])]
                blocks = [base[cols].reset_index(drop=True)]
                for n, fk, tc in kids:
                    t = Table(db.table_dict[n].df, fk, n, time_column=tc,
                              windows=windows, max_columns=max_cols)
                    blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                                  frame[tcol].to_numpy()).add_prefix(f"{n}__"))
                return pd.concat(blocks, axis=1)

            t0 = time.perf_counter()
            f_tr = build(train)
            f_va = build(val).reindex(columns=f_tr.columns, fill_value=np.nan)
            codes: dict = {}
            X, Xv = _numeric(f_tr, codes, True), _numeric(f_va, codes, False)
            assert_no_perfect_feature(X, y, list(f_tr.columns), context=f"{ds}/{value}")

            scores = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                rows = rng.choice(len(X), size=min(args.context, len(X)), replace=False)
                clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                                       random_state=seed,
                                       inference_config=NOAMP).fit(X[rows], y[rows])
                scores.append(roc_auc_score(y_va, clf.predict_proba(Xv)[:, 1]) * 100)
            m = float(np.mean(scores))
            table[(ds, value)] = m
            print(f"{ds:<10} {args.axis}={str(value):<6} VAL {m:6.2f} "
                  f"(sd {np.std(scores, ddof=1) if len(scores) > 1 else 0:.2f}, "
                  f"{f_tr.shape[1]} cols, {time.perf_counter() - t0:.0f}s)", flush=True)

    # Rank per task, then average. Ranks rather than raw AUC because the tasks sit at very
    # different levels -- a mean over 65 and 87 is dominated by whichever task moves most,
    # which is not what "robust across tasks" means.
    print(f"\n{'candidate':<12} " + " ".join(f"{ds:>10}" for ds, _ in TASKS)
          + "   mean rank", flush=True)
    ranks: dict = {v: [] for v in values}
    for ds, _ in TASKS:
        order = sorted(values, key=lambda v: -table.get((ds, v), -np.inf))
        for i, v in enumerate(order):
            ranks[v].append(i + 1)
    for v in values:
        cells = " ".join(f"{table.get((ds, v), float('nan')):>10.2f}" for ds, _ in TASKS)
        print(f"{str(v):<12} {cells}   {np.mean(ranks[v]):.2f}", flush=True)
    best = min(values, key=lambda v: np.mean(ranks[v]))
    print(f"\nDEFAULTROW\t{args.axis}\t{best}\t"
          f"mean rank {np.mean(ranks[best]):.2f}", flush=True)
    print("Chosen on validation across all four tasks. Test was never read, which is what "
          "makes this a default rather than a tuned setting.", flush=True)


if __name__ == "__main__":
    main()
