"""RelBench under an honest protocol: every setting chosen on validation.

The standing results table in ``STATUS.md`` reports, for each task, the best score over
several configurations -- `max_columns` on rel-event, join-versus-scan on rel-trial,
context size on rel-avito. Those configurations were compared *on the test split*, which
means the table is a per-task maximum selected on test, not a single procedure evaluated
once. It is not comparable with published single-configuration numbers, and it is
inflated by an unknown amount.

This script removes that. Configurations are compared on a validation split carved out
of training data; the test split is touched exactly once, at the end, with whatever the
validation comparison chose. Expect lower numbers than the hand-picked table. That is
the point of running it.

The settings swept are the three that have been measured to reverse across datasets:

    max_columns     +3.0 rel-event    0.0 rel-trial    -19.5 rel-f1
    context size    +0.4 rel-avito at 8.6%             -2.7 rel-trial at 23%
    categoricals    +1.25 rel-trial   0.0 rel-f1        -3.21 rel-event

No default serves all three, which is why they are calibrated rather than fixed.

Usage
-----
    python -m tabicl.scaling.eval_relbench_calibrated rel-trial study-outcome
    python -m tabicl.scaling.eval_relbench_calibrated rel-avito user-visits --children 3
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import Table, asof_statistics, sweep_configurations

WINDOWS = [pd.Timedelta(days=30), pd.Timedelta(days=365)]

# Each candidate is (max_columns, top_k_categories, include_mode). Deliberately small:
# the sweep costs one fit per candidate per task, and a wider grid would spend more
# compute on selection than the settings are worth.
#
# It was small in the wrong place, though. `(2, 4, True)` was the only cell with
# categorical blocks, which entangled three variables and pinned the categorical question
# to `max_columns=2` -- the budget later measured at 12.58 AUC *below* uncapped on rel-f1.
# So "the categorical blocks did not win" was never a statement about the categorical
# blocks. They now also appear at the budget that actually wins, and `mode` appears once
# without the histogram so the two are separable.
FEATURE_CANDIDATES = [
    (None, None, False),   # everything, no categorical blocks
    (4, None, False),
    (2, None, False),
    (2, 4, True),          # the categorical blocks, where they might pay
    (None, 4, True),       # ...and at the budget that wins, which was never tried
    (None, None, True),    # mode alone, to separate it from the histogram
]
CONTEXT_CANDIDATES: List[Optional[int]] = [5000, 10000, 20000, None]


def _numeric(df: pd.DataFrame, codes: Optional[dict] = None, fit: bool = True) -> np.ndarray:
    """Encode to float, with one codebook shared between the fit and eval frames.

    Factorizing each frame separately is not a smaller version of the same thing -- it
    changes what the columns *mean*. ``pd.factorize`` numbers values by order of first
    appearance, so a category is 3 in the fit frame and 7 in the eval frame, and the model
    is scored on a feature it was never trained on. This was found and fixed in
    ``eval_track_record``; the fix was never propagated here, so every configuration this
    sweep has ever compared was compared on scrambled categoricals.

    Unseen values map to -1, matching the sentinel ``pd.factorize`` already uses for nulls:
    "a value the fit set never showed me" is the honest encoding, and inventing a fresh
    code for it would be the same leak in a different direction.
    """
    if codes is None:
        codes = {}
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


def _stratified(frame: pd.DataFrame, target: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Keep ``n`` rows, preserving the label mix.

    rel-avito's positive rate is 0.905; an unstratified draw at 5,000 rows measures the
    draw rather than the setting.
    """
    if n >= len(frame):
        return frame
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in frame.groupby(frame[target], sort=True):
        take = max(1, int(round(n * len(group) / len(frame))))
        parts.append(group.iloc[rng.permutation(len(group))[:take]])
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _report_feasible(spec, predicted: int, budget: float) -> bool:
    ok = predicted <= budget
    if not ok:
        print(f"    {spec}: skipped, predicted {predicted / 2**30:.1f} GB "
              f"> budget {budget / 2**30:.1f} GB", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("task")
    parser.add_argument("--children", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--select-estimators", type=int, default=None,
                        help="ensemble size during selection. Defaults to --n-estimators, "
                             "and should stay there: a configuration's sign can flip with "
                             "ensemble size, so selecting at 1 and scoring at 4 chooses in a "
                             "regime that does not predict deployment. Measured on rel-event, "
                             "the categorical blocks are +2.87 at n_estimators=1 and -3.52 at "
                             "4. Lowering this is a real speed/soundness trade, not free.")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--memory-budget-gb", type=float, default=8.0,
                        help="predicted peak a feature spec may use before it is skipped")
    parser.add_argument("--select-context", type=int, default=5000,
                        help="context held fixed while feature specs are compared; the "
                             "feature sweep would otherwise run at full context, which "
                             "is quadratic and unaffordable on the large tasks")
    args = parser.parse_args()
    if args.select_estimators is None:
        args.select_estimators = args.n_estimators

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity_table = task.entity_col, task.target_col, task.entity_table

    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))

    entity_df = db.table_dict[entity_table].df
    pkey = db.table_dict[entity_table].pkey_col
    children = [
        (name, fk, tbl.time_col, len(tbl.df))
        for name, tbl in db.table_dict.items()
        for fk, points_to in (tbl.fkey_col_to_pkey_table or {}).items()
        if points_to == entity_table and tbl.time_col
    ]
    children.sort(key=lambda c: c[3])
    children = children[: args.children]
    print(f"{args.dataset}/{args.task}  train={len(train)} val={len(val)} test={len(test)}  "
          f"children={[c[0] for c in children]}", flush=True)

    def predicted_bytes(spec: Tuple) -> int:
        """Rough peak of the as-of scan for one feature spec.

        Deliberately a prediction rather than a try/except. The scan's prefix arrays are
        sized by the *child* table, not by the context, so shrinking the context does not
        make an expensive spec affordable -- rel-avito reached 24.6 GB on a 5,000-row
        context for exactly this reason. And an over-large allocation on a paging OS
        thrashes rather than raising, so there is nothing to catch.

        Counts float64 arrays over child rows: roughly four per numeric column (sum,
        sumsq, non-null count, plus the values re-read for min/max) and one per category
        bucket, with the window blocks re-reading the same columns.
        """
        max_cols, top_k, _use_mode = spec
        total = 0
        for name, fk, time_col, n_rows in children:
            frame = db.table_dict[name].df
            excluded = {fk, time_col}
            numeric = [c for c in frame.columns
                       if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
            categorical = [c for c in frame.columns
                           if c not in excluded and c not in numeric]
            if max_cols is not None:
                numeric = numeric[:max_cols]
                categorical = categorical[:max_cols]
            per_column = 4 * n_rows * 8
            total += len(numeric) * per_column * (1 + len(WINDOWS))
            if top_k:
                total += len(categorical) * (top_k + 1) * n_rows * 8 * (1 + len(WINDOWS))
        return total

    def build(frame: pd.DataFrame, spec: Tuple) -> pd.DataFrame:
        max_cols, top_k, use_mode = spec
        base = frame.merge(entity_df, left_on=key, right_on=pkey, how="left")
        drop = {target, key, pkey, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for name, fk, time_col, _ in children:
            table = Table(
                db.table_dict[name].df, fk, name, time_column=time_col, windows=WINDOWS,
                max_columns=max_cols, top_k_categories=top_k, include_mode=use_mode,
            )
            blocks.append(
                asof_statistics(table, frame[key].to_numpy(), frame[tcol].to_numpy())
                .add_prefix(f"{name}__")
            )
        return pd.concat(blocks, axis=1)

    def fit_score(fit_frame, eval_frame, spec, context, n_estimators) -> float:
        if context is not None:
            fit_frame = _stratified(fit_frame, target, context)
        f_fit = build(fit_frame, spec)
        f_eval = build(eval_frame, spec).reindex(columns=f_fit.columns, fill_value=np.nan)
        # AMP is on by default and costs 7.3 AUC on rel-event, so it is disabled for any
        # accuracy measurement. With it off, CUDA reproduces CPU exactly.
        cfg = None if args.device == "cpu" else {
            k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
        }
        model = TabICLClassifier(
            n_estimators=n_estimators, device=args.device, random_state=0,
            **({"inference_config": cfg} if cfg else {})
        ).fit(_numeric(f_fit, codes := {}, fit=True), fit_frame[target].to_numpy())
        proba = model.predict_proba(_numeric(f_eval, codes, fit=False))[:, 1]
        return roc_auc_score(eval_frame[target].to_numpy(), proba) * 100

    def safe(label, fn):
        """Score a candidate, treating "cannot run" as unselectable rather than fatal.

        rel-avito's full context is 116,598 rows and OOMs at 59 GB on CPU. That is a
        real property of the configuration, not an accident of the sweep, so the right
        response is to score it -inf and carry on -- the calibration then selects
        something that actually runs, which is what a caller wants.
        """
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001 - any failure is unselectable
            print(f"    {label}: unusable ({type(exc).__name__}: {str(exc)[:70]})", flush=True)
            return float("-inf")

    # ---- selection, on validation only -------------------------------------------
    t0 = time.perf_counter()
    # Feature specs are compared at a small, fixed context. Holding context constant is
    # what makes the comparison about features; letting it float would also make each
    # candidate cost a full-context fit, which on rel-avito's 116,598 rows is hours.
    feature_result = sweep_configurations(
        FEATURE_CANDIDATES,
        lambda spec: safe(spec, lambda: fit_score(
            train, val, spec, args.select_context, args.select_estimators)),
        tolerance=args.tolerance,
        # Fewer columns is cheaper, and FEATURE_CANDIDATES runs widest-first, so the
        # cheap end is the tail rather than the head.
        cheaper_first=False,
        feasible=lambda spec: _report_feasible(spec, predicted_bytes(spec),
                                               args.memory_budget_gb * 2**30),
    )
    print(f"\nfeature spec  chosen={feature_result.chosen}  {feature_result.curve}", flush=True)

    # Candidates at or above the pool size are the same experiment; running each would
    # refit identical data. rel-f1 has 1,353 training rows, so all four collapse to one.
    seen, context_candidates = set(), []
    for candidate in CONTEXT_CANDIDATES:
        size = len(train) if candidate is None else min(candidate, len(train))
        if size not in seen:
            seen.add(size)
            context_candidates.append(candidate)

    context_result = sweep_configurations(
        context_candidates,
        lambda n: safe(f"context={n}", lambda: fit_score(
            train, val, feature_result.chosen, n, args.select_estimators)),
        tolerance=args.tolerance,
        cheaper_first=True,
    )
    print(f"context       chosen={context_result.chosen}  {context_result.curve}", flush=True)
    print(f"selection took {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- the test split, once ------------------------------------------------------
    fit_frame = pd.concat([train, val], ignore_index=True)
    auc = fit_score(
        fit_frame, test, feature_result.chosen, context_result.chosen, args.n_estimators
    )
    print(
        f"\n{args.dataset}/{args.task}  CALIBRATED TEST ROC-AUC x100 = {auc:.2f}"
        f"   (max_columns={feature_result.chosen[0]}, top_k={feature_result.chosen[1]}, "
        f"mode={feature_result.chosen[2]}, context={context_result.chosen})",
        flush=True,
    )


if __name__ == "__main__":
    main()
