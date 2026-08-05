"""Does a trial's shared-key track record add to the pipeline on rel-trial?

rel-trial is the largest remaining gap: 66.50 against TabPFN-REL's 76.43. It is also the
one task where a foundation model beats the GNN, which points at featurisation rather than
graph structure, and it has proved insensitive to every lever tried -- column budget 0.0,
relation breadth +0.35, both inside the noise floor.

The candidate is the outcome history of trials sharing a sponsor, condition, facility or
intervention. Gated before being built (`key_target_history` docstring): each key scores
60-61 standalone test AUC at 76-88% coverage, against a whole pipeline at 66.50.

Three arms, and the middle one is not optional. On rel-event a neighbour-label feature
looked like a 74 AUC discovery and turned out to be mostly *degree* -- a count carrying no
outcome information at all. So the count columns get their own arm here, and the claim
"outcome history helps" only survives if the full block beats counts-only.

Nothing in `key_target_history` is rel-trial-specific -- it needs a link table, labels,
timestamps and a horizon -- so the runner takes the dataset and task as arguments. Use
`--gate` first on a new task: it prints each key's standalone AUC and coverage without
touching a GPU, and that ratio against the existing pipeline is what predicted the
difference between this working on rel-trial and the graph version failing on rel-event.

Usage
-----
    python -m tabicl.scaling.eval_track_record [dataset] [task] [--gate]
    python -m tabicl.scaling.eval_track_record rel-avito user-visits --gate
    python -m tabicl.scaling.eval_track_record --calibrated --seeds 5
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
from tabicl.scaling import Table, asof_statistics, key_target_history
from tabicl.scaling._leakage import permutation_test, temporal_control

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}

# Aggregation windows are task-scale, not universal: clinical trials run for years, ad
# impressions for days. A single default would quietly handicap one task or the other.
DEFAULT_WINDOWS = {
    "rel-trial": "365,1095",
    "rel-avito": "7,30",
    "rel-event": "30,365",
    "rel-f1": "365,1095",
}

# Printed beside a result so it is never read against the wrong task's numbers.
REFERENCE = {
    "rel-trial": "ours 69.36, TabPFN-REL 76.43, RelGNN 71.24, RDBLearn 72.89",
    "rel-event": "ours 78.11, TabPFN-REL 85.38, RelGNN 86.18, RDBLearn 73.70",
    "rel-avito": "ours 64.85, TabPFN-REL 66.68, RelGNN 66.18, RDBLearn 66.76",
    "rel-f1": "ours 80.70, TabPFN-REL 79.98, RelGNN 85.69, RDBLearn 82.72",
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
    ap.add_argument("--gate", action="store_true",
                    help="standalone AUC and coverage per key, no GPU. Run this on a new "
                         "task before building anything: a key worth less than the "
                         "existing pipeline has no room to help.")
    ap.add_argument("--window-days", default=None,
                    help="comma-separated aggregation windows; defaults per dataset")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--no-horizon", action="store_true",
                    help="ignore the 365-day resolution window. Wrong, and kept only to "
                         "measure what it is worth.")
    ap.add_argument("--static-links", action="store_true",
                    help="ignore link-table timestamps, counting memberships that formed "
                         "after the cutoff. Wrong; kept to measure what the causal "
                         "filtering is worth, which on rel-event is the whole result.")
    ap.add_argument("--calibrated", action="store_true",
                    help="choose the arm and the context size on validation, then score "
                         "test once. A paired A/B is not eligible for the headline table; "
                         "this is.")
    args = ap.parse_args()

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    horizon = None if args.no_horizon else getattr(task, "timedelta", None)
    y, y_te = train[target].to_numpy(), test[target].to_numpy()
    spec = args.window_days or DEFAULT_WINDOWS.get(args.dataset, "30,365")
    WINDOWS = [pd.Timedelta(days=int(d)) for d in spec.split(",")]
    print(f"{args.dataset}/{args.task}  horizon={horizon}  windows={spec}  "
          f"train={len(train)} test={len(test)}", flush=True)

    # --- base features -------------------------------------------------------------------
    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
            for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
            if pt == entity and t.time_col][:3]

    def build_base(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS, max_columns=2)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        return pd.concat(blocks, axis=1)

    # --- which keys does this schema even offer? ----------------------------------------
    link_specs = []
    for name, tbl in db.table_dict.items():
        for fk, pt in (tbl.fkey_col_to_pkey_table or {}).items():
            if pt != entity:
                continue
            for other in (tbl.fkey_col_to_pkey_table or {}):
                if other != fk:
                    # Qualify by table: the same key name appears in several link tables
                    # (rel-event has `event` twice, rel-f1 `raceId` three times) and an
                    # unqualified prefix silently collides the blocks on concat.
                    short = f"{name}_{other}".replace("_id", "").replace("_ID", "")
                    tc = tbl.time_col if not args.static_links else None
                    cols = [fk, other] + ([tc] if tc else [])
                    link_specs.append((short, tbl.df[cols], fk, other, tc))
    timed = [s[0] for s in link_specs if s[4]]
    untimed = [s[0] for s in link_specs if not s[4]]
    print(f"candidate keys: {[s[0] for s in link_specs] or 'NONE'}", flush=True)
    print(f"  link tables WITH timestamps (causal): {timed or 'none'}", flush=True)
    if untimed:
        print(f"  link tables WITHOUT timestamps: {untimed} -- memberships formed after a "
              f"cutoff are visible to it, so any lift from those keys is an UPPER BOUND",
              flush=True)
    if not link_specs:
        print("no table links two entities of this type -- this feature cannot be built "
              "on this task", flush=True)
        return

    if args.gate:
        # Standalone AUC per key, before any base features or GPU work. A key worth less
        # than the pipeline it must improve has no room; that ratio is what separated
        # rel-trial (61 against 66.5, worked) from rel-event's graph version (68 against
        # 83, did not).
        val = task.get_table("val", mask_input_cols=False).df
        print(f"\n{'key':<20} {'coverage':>9} {'val AUC':>9} {'test AUC':>9}", flush=True)
        for short, frame, fk, other, ltc in link_specs:
            row = []
            for split in (val, test):
                block = key_target_history(
                    frame[[fk, other]], label_entities=train[key].to_numpy(),
                    label_values=y, label_times=train[tcol].to_numpy(),
                    query_entities=split[key].to_numpy(),
                    query_times=split[tcol].to_numpy(), label_horizon=horizon,
                    link_times=frame[ltc].to_numpy() if ltc else None,
                )
                rate = block["hist__positive_rate"].to_numpy()
                truth = split[target].to_numpy()
                cov = float(np.mean(~np.isnan(rate)))
                filled = np.where(np.isnan(rate), y.mean(), rate)
                auc = (roc_auc_score(truth, filled) * 100
                       if len(np.unique(truth)) > 1 else float("nan"))
                row.append((cov, auc))
            print(f"{short:<20} {row[0][0]:>9.3f} {row[0][1]:>9.2f} {row[1][1]:>9.2f}",
                  flush=True)
        print("\nCompare against the task's existing calibrated number before building.",
              flush=True)
        return

    b_tr = build_base(train)
    b_te = build_base(test).reindex(columns=b_tr.columns, fill_value=np.nan)

    def track(entities, times, labels, shift=None):
        stamps = np.asarray(times)
        if shift:
            stamps = stamps - pd.Timedelta(days=shift)
        blocks = []
        for short, frame, fk, other, ltc in link_specs:
            blocks.append(key_target_history(
                frame[[fk, other]], label_entities=train[key].to_numpy(),
                label_values=labels, label_times=train[tcol].to_numpy(),
                query_entities=entities, query_times=stamps,
                label_horizon=horizon, prefix=f"{short}__",
                link_times=frame[ltc].to_numpy() if ltc else None,
            ))
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=range(len(entities)))

    t_tr = track(train[key].to_numpy(), train[tcol].to_numpy(), y)
    t_te = track(test[key].to_numpy(), test[tcol].to_numpy(), y)
    count_cols = [c for c in t_tr.columns if c.endswith("n_prior")]
    struct_cols = [c for c in t_tr.columns if c.endswith("n_linked")]
    print(f"track-record blocks: {list(t_tr.columns)}", flush=True)

    # --- controls, before any comparison -------------------------------------------------
    def rate_block(labels, shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), labels, shift=shift)
        return block[[c for c in block.columns if c.endswith("positive_rate")]].mean(axis=1)

    def rate_only_score(labels, shift=None):
        rate = rate_block(labels, shift=shift)
        return roc_auc_score(y_te, rate.fillna(np.nanmean(labels)).to_numpy())

    print("\ncontrol 1: permutation (null = permuted distribution)", flush=True)
    perm = permutation_test(rate_only_score, y, n_permutations=5, n_sigma=3.0)
    print(f"  {perm!r}", flush=True)

    # Shifts must be scaled to the task, not fixed in days. A shift far larger than the
    # task's own time span removes *every* usable label, the feature goes constant, and
    # the control passes at exactly 0.5 having tested nothing -- which is what a hardcoded
    # 180/365 days did on rel-event, whose horizon is 7 days. Coverage is printed so a
    # vacuous pass is visible rather than reassuring.
    span_days = float((train[tcol].max() - train[tcol].min()) / pd.Timedelta(days=1))
    shifts = (0.0, round(0.05 * span_days), round(0.15 * span_days))
    print(f"control 2: temporal (span {span_days:.0f}d, shifts {shifts[1:]}d)", flush=True)
    base_cov = float(rate_block(y).notna().mean())
    for s in shifts[1:]:
        cov = float(rate_block(y, shift=s).notna().mean())
        print(f"  coverage at -{s:.0f}d: {cov:.3f} (unshifted {base_cov:.3f})", flush=True)
        if base_cov > 0 and cov < 0.1 * base_cov:
            print("  *** control is VACUOUS at this shift -- nearly all labels removed, "
                  "so a pass proves nothing", flush=True)
    temporal = temporal_control(lambda days: rate_only_score(y, shift=days), shifts=shifts)
    print(f"  {temporal!r}", flush=True)

    # The counts arm needs its own temporal control, and it is the one a permutation test
    # cannot cover: counts do not depend on label *values*, so shuffling leaves them
    # unchanged. Reporting a counts-only result on the strength of a rate-only control --
    # which is what happened on rel-event first time round -- controls nothing.
    def count_only_score(shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), y, shift=shift)
        counts = block[[c for c in block.columns if c.endswith("n_prior")]].sum(axis=1)
        return roc_auc_score(y_te, counts.to_numpy())

    print("control 3: temporal, on the COUNT columns", flush=True)
    temporal_counts = temporal_control(lambda days: count_only_score(shift=days),
                                       shifts=shifts)
    print(f"  {temporal_counts!r}", flush=True)

    # And on the structural column, which is the arm validation actually keeps choosing.
    # Controls 1-3 all test columns the `+struct` arm does not contain, so passing them
    # says nothing about it -- reporting a struct-only result on their strength would be
    # the same mistake as reporting a counts-only result on a rate-only control.
    def struct_only_score(shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), y, shift=shift)
        deg = block[[c for c in block.columns if c.endswith("n_linked")]].sum(axis=1)
        return roc_auc_score(y_te, deg.to_numpy())

    print("control 4: temporal, on the STRUCTURAL column (n_linked)", flush=True)
    temporal_struct = temporal_control(lambda days: struct_only_score(shift=days),
                                       shifts=shifts)
    print(f"  {temporal_struct!r}", flush=True)
    if not temporal_struct.passed:
        print("  *** n_linked reaches past the cutoff. Where a link table carries no "
              "timestamp this is expected and unfixable -- exclude those keys rather than "
              "reporting the arm.", flush=True)

    # And the question that decides whether the counts are even a label feature: pure
    # structural degree consults no labels at all, so if it scores alike there is no
    # leakage question to answer.
    struct = t_te[struct_cols].sum(axis=1).to_numpy()
    prior = t_te[count_cols].sum(axis=1).to_numpy()
    print(f"  structural degree alone (no labels): "
          f"{roc_auc_score(y_te, struct) * 100:.2f}", flush=True)
    print(f"  resolved-label counts alone:         "
          f"{roc_auc_score(y_te, prior) * 100:.2f}", flush=True)
    # Each arm is gated by the controls that test *its own* columns, so one failing family
    # does not block a clean one -- and, more importantly, a passing family cannot vouch
    # for an arm it never touched.
    ok = {"base": True,
          "+struct": temporal_struct.passed,
          "+counts": temporal_counts.passed,
          "+history": perm.passed and temporal.passed and temporal_counts.passed
                      and temporal_struct.passed}
    print(f"\narm eligibility: {ok}", flush=True)
    if not any(v for k, v in ok.items() if k != "base"):
        print("CONTROLS FAILED for every feature arm -- nothing to measure", flush=True)
        return
    for name, passed in ok.items():
        if not passed:
            print(f"  {name} is EXCLUDED from selection: its own controls failed", flush=True)

    # --- three arms ----------------------------------------------------------------------
    def stack(base, block, cols=None):
        chosen = block if cols is None else block[cols]
        return _numeric(pd.concat([base.reset_index(drop=True),
                                   chosen.reset_index(drop=True)], axis=1))

    arms = {
        "base": (_numeric(b_tr), _numeric(b_te)),
        "+struct": (stack(b_tr, t_tr, struct_cols), stack(b_te, t_te, struct_cols)),
        "+counts": (stack(b_tr, t_tr, count_cols), stack(b_te, t_te, count_cols)),
        "+history": (stack(b_tr, t_tr), stack(b_te, t_te)),
    }
    # An arm whose own controls failed is not offered to validation at all. Selection
    # cannot be allowed to pick a leaking arm and have the protocol launder it.
    arms = {k: v for k, v in arms.items() if ok.get(k, True)}
    print(f"{arms['base'][0].shape[1]} base -> {arms['+history'][0].shape[1]} with history, "
          f"{len(arms['base'][0])} train rows, context={args.context}", flush=True)

    def score(X, Xe, rows, seed, truth=None):
        clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                               random_state=seed, inference_config=NOAMP).fit(X[rows], y[rows])
        target_y = y_te if truth is None else truth
        return roc_auc_score(target_y, clf.predict_proba(Xe)[:, 1]) * 100

    if args.calibrated:
        val = task.get_table("val", mask_input_cols=False).df
        y_va = val[target].to_numpy()
        b_va = build_base(val).reindex(columns=b_tr.columns, fill_value=np.nan)
        t_va = track(val[key].to_numpy(), val[tcol].to_numpy(), y)
        val_arms = {k: v for k, v in {
            "base": _numeric(b_va),
            "+struct": stack(b_va, t_va, struct_cols),
            "+counts": stack(b_va, t_va, count_cols),
            "+history": stack(b_va, t_va),
        }.items() if k in arms}
        results = []
        for seed in range(args.seeds):
            best = None
            print(f"\n-- seed {seed}: selection (validation only) --", flush=True)
            n_train = len(arms["base"][0])
            grid = sorted({max(1000, n_train // 4), max(2000, n_train // 2), n_train})
            for name in arms:
                for size in grid:
                    n = len(arms[name][0])
                    rows = np.random.default_rng(seed).choice(n, size=min(size, n),
                                                              replace=False)
                    v = score(arms[name][0], val_arms[name], rows, seed, truth=y_va)
                    print(f"  {name:<9} context={size:<6} val={v:.2f}", flush=True)
                    if best is None or v > best[0]:
                        best = (v, name, size)
            val_auc, name, size = best
            n = len(arms[name][0])
            rows = np.random.default_rng(seed).choice(n, size=min(size, n), replace=False)
            auc = score(arms[name][0], arms[name][1], rows, seed)
            results.append((auc, name, size, val_auc))
            print(f"  chosen {name} context={size} -> VAL {val_auc:.2f}  TEST {auc:.2f}",
                  flush=True)

        aucs = np.array([r[0] for r in results])
        vals = np.array([r[3] for r in results])
        chose = [r[1] for r in results]
        if all(c == "+counts" for c in chose):
            print("\n*** WARNING: validation chose the counts-only arm, but the controls "
                  "above were run on the positive-rate columns. The reported arm is "
                  "UNCONTROLLED -- a permutation test is meaningless for counts (they do "
                  "not depend on label values), so what this needs is a temporal control "
                  "on the count columns specifically.", flush=True)
        print(f"\n{args.dataset}/{args.task}  CALIBRATED TEST ROC-AUC x100 = {aucs.mean():.2f} "
              f"+- {aucs.std(ddof=1) if len(aucs) > 1 else 0:.2f} over {len(aucs)} "
              f"replicates (range {aucs.min():.2f}-{aucs.max():.2f})", flush=True)
        # Machine-readable line so a table can be assembled across tasks without re-running.
        print(f"TABLEROW\t{args.dataset}/{args.task}\t{len(train)}\t{len(val)}\t{len(test)}"
              f"\t{vals.mean():.2f}\t{vals.std(ddof=1) if len(vals) > 1 else 0:.2f}"
              f"\t{aucs.mean():.2f}\t{aucs.std(ddof=1) if len(aucs) > 1 else 0:.2f}"
              f"\t{max(set(chose), key=chose.count)}", flush=True)
        print(f"validation chose: {chose}; sizes {[r[2] for r in results]}", flush=True)
        print(f"reference: {REFERENCE.get(args.dataset, 'see PERFORMANCE.md')}", flush=True)
        return

    header = "".join(f"{name:>10}" for name in arms)
    print(f"\n{'seed':>5}{header}", flush=True)
    results = {k: [] for k in arms}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        n = len(arms["base"][0])
        rows = rng.choice(n, size=min(args.context, n), replace=False)
        t0 = time.perf_counter()
        for name, (X, Xe) in arms.items():
            results[name].append(score(X, Xe, rows, seed))
        cells = "".join(f"{results[name][-1]:>10.2f}" for name in arms)
        print(f"{seed:>5}{cells}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Every contrast that exists, paired by seed. Naming both sides keeps a reader from
    # attributing a gap to the wrong difference, which is this project's recurring error.
    pairs = [("+history", "base"), ("+history", "+counts"), ("+counts", "+struct"),
             ("+struct", "base"), ("+history", "+struct")]
    for hi, lo in pairs:
        if hi not in results or lo not in results:
            continue
        g = np.array(results[hi]) - np.array(results[lo])
        print(f"{hi} over {lo}: mean {g.mean():+.2f} sd "
              f"{g.std(ddof=1) if len(g) > 1 else 0:.2f} over {len(g)} seeds, "
              f"{(g > 0).sum()}/{len(g)} positive", flush=True)
    print(f"base mean {np.mean(results['base']):.2f}, "
          f"+history mean {np.mean(results['+history']):.2f}  "
          f"({REFERENCE.get(args.dataset, 'see PERFORMANCE.md')})", flush=True)
    print("NOTE: the +-0.6 floor applies.", flush=True)


if __name__ == "__main__":
    main()
