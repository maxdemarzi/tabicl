"""Is our base relational pipeline itself the thing holding the numbers down?

Evidence that it might be. On rel-trial our unaugmented base arm scores 63.82 while
Rel-LLM reports a LightGBM on the entity table alone at 70.09. Even allowing that our
entity-only reproduction is not yet faithful, a relational pipeline that loses to raw
entity columns is not being limited by its *features* — it is being limited by how they
are built. Every headline number on this branch sits on top of that base, so a point here
is a point everywhere, and it is cheaper than another feature family.

Four knobs, swept one at a time against a fixed reference so no comparison ever moves two
variables:

  ``max_columns``   already known to span 22 points across tasks (+3.0 / 0.0 / −19.5)
  ``n_estimators``  already known to flip the sign of the categorical blocks
  child count       "which relations you traverse" was measured at +0.083; retest properly
  windows           +0.089 on rel-f1, ~0 elsewhere — but never swept per task

Selection is on **validation**; test is not touched at all here. The point is to find a
better base to hand to the calibrated runners, not to produce a reportable number.

Usage
-----
    python -m tabicl.scaling.eval_base_sweep rel-trial study-outcome
"""

from __future__ import annotations

import argparse
import itertools
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
WINDOW_CHOICES = {
    "short": (7, 30),
    "medium": (30, 365),
    "long": (365, 1095),
}


def _numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    y, y_va = train[target].to_numpy(), val[target].to_numpy()

    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    all_kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
                for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                if pt == entity and t.time_col]
    print(f"{args.dataset}/{args.task}: {len(all_kids)} timestamped child tables, "
          f"train={len(train)} val={len(val)}", flush=True)

    def build(frame, max_columns, n_kids, windows):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in all_kids[:n_kids]:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc,
                      windows=[pd.Timedelta(days=d) for d in windows],
                      max_columns=max_columns)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    cache: dict = {}

    def evaluate(max_columns, n_kids, windows, n_estimators):
        spec = (max_columns, n_kids, windows)
        if spec not in cache:
            f_tr = build(train, *spec)
            f_va = build(val, *spec).reindex(columns=f_tr.columns, fill_value=np.nan)
            cache[spec] = (_numeric(f_tr), _numeric(f_va), f_tr.shape[1])
        X, Xva, width = cache[spec]
        scores = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            rows = rng.choice(len(X), size=min(args.context, len(X)), replace=False)
            clf = TabICLClassifier(n_estimators=n_estimators, device=args.device,
                                   random_state=seed,
                                   inference_config=NOAMP).fit(X[rows], y[rows])
            scores.append(roc_auc_score(y_va, clf.predict_proba(Xva)[:, 1]) * 100)
        return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, width

    # Reference point: what the runners currently use.
    ref = (2, min(3, len(all_kids)), WINDOW_CHOICES["medium"], 4)
    print(f"\nreference (current default): max_columns={ref[0]} kids={ref[1]} "
          f"windows={ref[2]} n_estimators={ref[3]}", flush=True)
    t0 = time.perf_counter()
    ref_score, ref_sd, ref_width = evaluate(*ref)
    print(f"  val {ref_score:.2f} +- {ref_sd:.2f}  ({ref_width} features, "
          f"{time.perf_counter() - t0:.0f}s)\n", flush=True)

    sweeps = {
        "max_columns": [(v, ref[1], ref[2], ref[3]) for v in (2, 4, 8, None)],
        "n_estimators": [(ref[0], ref[1], ref[2], v) for v in (1, 2, 4, 8)],
        "child tables": [(ref[0], v, ref[2], ref[3])
                         for v in sorted({1, 3, min(6, len(all_kids)), len(all_kids)})],
        "windows": [(ref[0], ref[1], w, ref[3]) for w in WINDOW_CHOICES.values()],
    }

    best = (ref_score, ref)
    for knob, settings in sweeps.items():
        print(f"-- {knob} --", flush=True)
        for spec in settings:
            score, sd, width = evaluate(*spec)
            delta = score - ref_score
            flag = "  <-- best so far" if score > best[0] else ""
            if score > best[0]:
                best = (score, spec)
            print(f"  {str(spec):<40} val {score:>6.2f} +- {sd:.2f} "
                  f"({width:>4} feat) {delta:>+6.2f}{flag}", flush=True)

    print(f"\nbest on validation: {best[1]} at {best[0]:.2f}, "
          f"{best[0] - ref_score:+.2f} over the current default", flush=True)
    print("Chosen on validation only; test was never touched here. Feed the winner to "
          "eval_track_record / eval_relbench_calibrated for a reportable number.",
          flush=True)
    print("NOTE: +-0.6 floor; anything smaller is a tie and the cheaper setting wins.",
          flush=True)


if __name__ == "__main__":
    main()
