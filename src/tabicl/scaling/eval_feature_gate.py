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
from tabicl.scaling._guards import assert_no_perfect_feature
from tabicl.scaling._calendar import calendar_features

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
    # Narrower, not wider: `max_columns` has never bounded the nunique block, so it has
    # been bounding numeric columns only while every categorical column emitted a
    # distinct-count regardless. This is the only variant here that can *remove* columns,
    # and the project's own column-budget measurements say narrow feature sets win.
    "narrow": dict(budget_categoricals=True),
    # The timestamp is the one column never aggregated, so nothing emitted so far says
    # *when*. On a task asking whether a user acts in the next four days, "days since last
    # activity" is the feature a practitioner reaches for first.
    "timing": dict(time_deltas=True),
    "all": dict(top_k_categories=4, include_mode=True, numeric_booleans=True),
}

# Variants that add a block to the entity frame rather than to a child Table. Both runners
# drop every datetime column when assembling that frame, the cutoff included, so nothing
# downstream can tell a Monday from a Saturday -- on tasks with four- and seven-day
# horizons. `calendar-trend` is separate because days-since-origin is monotone and every
# test row lies beyond the training range on it; cyclical features have no such problem.
FRAME_VARIANTS = {
    "calendar": dict(trend=False),
    "calendar-trend": dict(trend=True),
}

# Variants that change the MODEL rather than the features. Same frames on both sides, so
# the empty-block refusal does not apply -- there is nothing for them to make empty.
#
# These were added because the exhausted list covers ensemble size and context size but not
# how the ensemble normalizes, and the reasoning was that this pipeline emits counts, sums
# and rates -- heavy-tailed by construction, which is the distribution `quantile` exists
# for -- while `outlier_threshold=4.0` clips |z| > 4, which on a power-law count column is
# the tail rather than an outlier.
#
# **Measured 2026-08-05, and that reasoning was wrong.** On rel-trial: norm-quantile -3.07,
# norm-all -1.73, norm-robust -2.08, every one 0/5 positive and every one clearing the
# floor downwards. outliers-wide -0.73 and outliers-off -0.63, also 0/5 -- so clipping at
# 4.0 is doing real work rather than truncating signal. rel-avito agrees in direction and
# is null in size. The library defaults are right for this data.
#
# Kept for the record and because they are cheap to re-run, not because they are promising.
MODEL_VARIANTS = {
    "norm-quantile": dict(norm_methods=["none", "quantile"]),
    "norm-all": dict(norm_methods=["none", "power", "quantile", "robust"]),
    "norm-robust": dict(norm_methods=["none", "robust"]),
    "outliers-wide": dict(outlier_threshold=12.0),
    "outliers-off": dict(outlier_threshold=1e9),
}

# Variants that change WHICH training rows become context. Also same features both arms.
#
# The context is drawn uniformly at random from train, which is a choice nobody made. The
# selection thread ended by diagnosing the train-to-test gap as *temporal*: resampling
# improved coverage of the training pool and gained ~7.5 on every criterion scored on
# held-out train rows while test fell 1.44, because a later period is not the same
# distribution. A recency-weighted context attacks that diagnosis directly instead of
# working around it.
#
# And unlike resampling, this one is selectable. RelBench splits are temporal -- train
# then val then test -- so validation is itself a later period than train and can see a
# recency effect. What could not judge resampling was cross-validation over held-out
# *train* rows, which is a different instrument from the validation split.
CONTEXT_VARIANTS = {
    "recent": "the most recent rows by timestamp",
    "recent-half": "uniform within the most recent half of train",
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
    ap.add_argument("--variant", default="categories",
                    help="comma-separated. One build per distinct feature set, shared "
                         "across every variant that needs it, and one base arm scored per "
                         "seed and reused -- so the gaps are paired against the same "
                         "numbers and comparable to each other. Available: "
                         + ", ".join(sorted(set(VARIANTS) | set(MODEL_VARIANTS)
                                            | set(CONTEXT_VARIANTS) | set(FRAME_VARIANTS))))
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
    unknown = [v.strip() for v in args.variant.split(",")
               if v.strip() and v.strip() not in set(VARIANTS) | set(MODEL_VARIANTS)
               | set(CONTEXT_VARIANTS) | set(FRAME_VARIANTS)]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}")
    print(f"{args.dataset}/{args.task}  variants={args.variant}  windows={spec}  "
          f"max_columns={max_cols}  children={[k[0] for k in kids]}", flush=True)

    # Trend is measured from the training minimum so train and test share a scale; taking
    # each frame's own minimum would silently reset the origin at test time.
    train_origin = pd.Timestamp(train[tcol].min())

    def build(frame, extra, calendar=None):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        if calendar is not None:
            blocks.append(calendar_features(frame[tcol].to_numpy(),
                                            origin=train_origin, **calendar))
        for n, fk, tc in kids:
            table = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=windows,
                          max_columns=max_cols, min_category_share=args.category_share,
                          **extra)
            blocks.append(asof_statistics(table, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    # One build per distinct FEATURE set, shared across every variant that needs it. The
    # model-side and context-side variants all run on the base frames, so testing five of
    # them used to mean building the base frames five times -- and on rel-event a build is
    # the expensive half of the job.
    wanted = [v.strip() for v in args.variant.split(",") if v.strip()]
    frames, widths = {}, {}

    def ensure(label, extra, calendar=None):
        if label in frames:
            return
        t0 = time.perf_counter()
        f_tr = build(train, extra, calendar)
        f_te = build(test, extra, calendar).reindex(columns=f_tr.columns, fill_value=np.nan)
        codes: dict = {}
        frames[label] = (_numeric(f_tr, codes, True), _numeric(f_te, codes, False))
        widths[label] = list(f_tr.columns)
        assert_no_perfect_feature(frames[label][0], y, list(f_tr.columns), context=label)
        print(f"{label:>13}: {f_tr.shape[1]} features in {time.perf_counter() - t0:.0f}s",
              flush=True)

    ensure("base", {})

    def score(X, Xe, rows, seed, extra=None):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP,
                               **(extra or {})).fit(X[rows], y[rows])
        return roc_auc_score(y_te, clf.predict_proba(Xe)[:, 1]) * 100

    # Train row order by time, for the context variants. Ties keep their original order so
    # the choice is reproducible.
    train_order = np.argsort(train[tcol].to_numpy(), kind="stable")

    def context_rows(rng, n, size, variant):
        if variant == "recent":
            # Deterministic given the size, so its only seed-to-seed variation is the
            # model's -- which is the point: if this helps, it helps without a draw.
            return train_order[-size:]
        if variant == "recent-half":
            pool = train_order[len(train_order) // 2:]
            return rng.choice(pool, size=min(size, len(pool)), replace=False)
        return rng.choice(n, size=size, replace=False)

    # The base arm is scored once per seed and reused by every variant. Each variant is
    # therefore paired against the *same* base numbers, which is what makes the gaps
    # comparable to each other and not just each to zero.
    n = len(frames["base"][0])
    size = min(args.context, n)
    draws = {seed: np.random.default_rng(seed).choice(n, size=size, replace=False)
             for seed in range(args.seeds)}
    base_auc = {seed: score(*frames["base"], draws[seed], seed) for seed in range(args.seeds)}
    print(f"\nbase: {', '.join(f'{base_auc[s]:.2f}' for s in sorted(base_auc))}", flush=True)
    summary: list[str] = []

    for variant in wanted:
        model_side = variant in MODEL_VARIANTS
        context_side = variant in CONTEXT_VARIANTS
        if model_side or context_side:
            label = "base"
            detail = MODEL_VARIANTS[variant] if model_side else CONTEXT_VARIANTS[variant]
            print(f"\n{variant}: {'model' if model_side else 'context'}-side, identical "
                  f"features both arms ({len(widths['base'])} columns): {detail}", flush=True)
            n_changed = 0
            if context_side and size >= n:
                # The context already covers every training row, so "the most recent
                # `size`" is the same set -- and the gap would be pure model noise
                # reported as a measurement of recency. Same silent-null shape as an
                # empty feature block.
                print(f"{variant}: SKIPPED -- context {size} covers all {n} training "
                      f"rows, so this selects the same set as the base arm and any gap "
                      f"would be noise. Lower --context to measure it.", flush=True)
                continue
        else:
            label = variant
            if variant in FRAME_VARIANTS:
                ensure(label, {}, FRAME_VARIANTS[variant])
            else:
                ensure(label, VARIANTS[variant])
            # An empty block does not raise, it reports +0.00 with sd 0.00 --
            # indistinguishable from a clean null. Depth-2 spent a week "measured at no
            # effect" that way. Two of these have their own silent-empty path:
            # `min_category_share` can reject every column on a free-text schema, and a
            # schema with no boolean columns leaves `numeric_booleans` a no-op.
            added = [c for c in widths[label] if c not in set(widths["base"])]
            removed = [c for c in widths["base"] if c not in set(widths[label])]
            n_changed = len(added) + len(removed)
            if not n_changed:
                print(f"\n{variant}: SKIPPED -- changed no columns ({len(widths['base'])} "
                      f"either way), so any gap would be an artefact of an empty block. "
                      f"Either this schema has nothing for it to act on, or the "
                      f"--category-share gate ({args.category_share}) rejected every "
                      f"column.", flush=True)
                continue
            print(f"\n{variant}: +{len(added)} columns, -{len(removed)}; "
                  f"e.g. {(added or removed)[:3]}", flush=True)

        print(f"{'seed':>5} {'base':>9} {variant:>14} {'gap':>8}", flush=True)
        gaps = []
        for seed in range(args.seeds):
            rows = (context_rows(np.random.default_rng(seed), n, size, variant)
                    if context_side else draws[seed])
            b = score(*frames[label], rows, seed,
                      MODEL_VARIANTS[variant] if model_side else None)
            gaps.append(b - base_auc[seed])
            print(f"{seed:>5} {base_auc[seed]:>9.2f} {b:>14.2f} {gaps[-1]:>+8.2f}", flush=True)

        g = np.array(gaps)
        sd = g.std(ddof=1) if len(g) > 1 else 0.0
        # Report the standard error, not only the spread. The ±0.6 floor is about a
        # *paired gap on these tasks*, and how far a mean must sit from zero to clear it
        # depends on the seed count: rel-event's paired sd is around 1.4, so five seeds
        # give SE 0.63 and a gap there needs roughly 1.3 to be readable, while rel-trial's
        # 0.44 gives SE 0.20. Two gaps of +0.5 on different tasks are not the same result,
        # and the sd column alone invites reading them as though they were.
        se = sd / np.sqrt(len(g)) if len(g) > 1 else 0.0
        verdict = "clears" if abs(g.mean()) > max(0.6, 2 * se) else "inside noise"
        row = (f"GATEROW\t{args.dataset}/{args.task}\t{variant}\t{g.mean():+.2f}\t"
               f"{sd:.2f}\tSE {se:.2f}\t{(g > 0).sum()}/{len(g)}\t{n_changed}\t{verdict}")
        summary.append(row)
        print(row, flush=True)

    # Every row again, together, at the end. A caller that pipes this through `tail`
    # otherwise keeps only the last variant or two -- which is exactly what happened:
    # `tail -60` silently discarded timing, calendar, booleans and narrow on every task in
    # a run, and the results were computed and thrown away rather than missing loudly.
    print(f"\n===== {args.dataset}/{args.task}: {len(summary)} of {len(wanted)} variants "
          f"measured", flush=True)
    for row in summary:
        print(row, flush=True)
    skipped = [v for v in wanted if not any(f"\t{v}\t" in r for r in summary)]
    if skipped:
        print(f"SKIPPED (nothing to measure): {', '.join(skipped)}", flush=True)
    print("Gate only. +-0.6 floor, and this is test-side -- a pass earns a calibrated "
          "run, not a table entry.", flush=True)


if __name__ == "__main__":
    main()
