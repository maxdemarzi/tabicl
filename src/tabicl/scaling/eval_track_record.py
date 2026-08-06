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
from tabicl.scaling._guards import assert_no_perfect_feature
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


_CATEGORY_MAPS: dict = {}


def _numeric(df: pd.DataFrame, fit: bool = False, tag: str = "") -> np.ndarray:
    """Encode a feature frame, with categorical codes CONSISTENT ACROSS SPLITS.

    The previous version called ``pd.factorize`` on each frame independently. Factorize
    assigns codes by order of first appearance, so the same category received *different
    integers in train and in test* -- a value coded 3 for fitting could be 7 at prediction
    time. Every categorical column was therefore not merely arbitrarily ordered but
    inconsistently ordered, which is worse: the model learns a mapping that does not hold
    where it is applied.

    Categories are now learned on the fitting frame and reused. Unseen values encode to -1,
    which is a distinguishable "not in training" rather than a collision with a real code.
    """
    out = df.copy()
    # Key by the frame's own column set, so two arms with different columns cannot share a
    # map and no caller has to remember to pass a distinct tag.
    tag = tag or str(hash(tuple(df.columns)))
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        keyed = f"{tag}:{col}"
        if fit or keyed not in _CATEGORY_MAPS:
            codes, uniques = pd.factorize(out[col])
            _CATEGORY_MAPS[keyed] = {v: i for i, v in enumerate(uniques)}
            out[col] = codes
        else:
            out[col] = out[col].map(_CATEGORY_MAPS[keyed]).fillna(-1)
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
    ap.add_argument("--stratify-context", action="store_true",
                    help="draw each resampled context in proportion to the training class "
                         "balance. Only meaningful with --resample > 1: it removes the "
                         "class-balance wobble between draws, which is noise added to the "
                         "quantity resampling exists to average down.")
    ap.add_argument("--cv-time-ordered", action="store_true",
                    help="forward-chaining folds: each is scored using only earlier rows. "
                         "Required for this benchmark -- random k-folds train on the future "
                         "to predict the past, and did exactly that, rewarding resampling "
                         "with +5.3 of CV score while test fell 0.76.")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="select on k-fold CV over train instead of the single validation "
                         "split. 0 keeps the current behaviour. The val splits here are "
                         "588-2013 rows and have twice blocked a real gain: child count was "
                         "undecidable (0.61 spread across every setting) and per-key "
                         "selection ranked options opposite to test. Selection noise, not "
                         "feature quality, is the binding constraint.")
    ap.add_argument("--resample", type=int, default=1,
                    help="average predictions over N independent context draws. 1 is the "
                         "existing behaviour exactly. Attacks the variance that dominates "
                         "these tasks: rel-event's replicates span 10.8 points on the "
                         "context draw alone. Distinct from --n-estimators, which varies "
                         "the model seed at a fixed context and measured at nothing.")
    ap.add_argument("--children", type=int, default=3,
                    help="how many timestamped child tables to aggregate. The default of 3 "
                         "was never chosen -- it is a hardcoded slice, and the tables it "
                         "keeps are whichever come first in dictionary order, out of ten on "
                         "rel-trial. The base sweep says breadth matters: one child costs 30 "
                         "points on rel-event and 8 on rel-avito. 0 means all.")
    ap.add_argument("--text", action="store_true",
                    help="add a TF-IDF + SVD embedding of the entity table's free-text "
                         "columns as its own arm. RESEARCH 6f: our pipeline has never used "
                         "text content, and on rel-trial four columns score 60.2-64.1 "
                         "standalone against a 69.36 pipeline.")
    ap.add_argument("--text-components", type=int, default=32)
    ap.add_argument("--text-rows", type=int, default=20,
                    help="child rows per entity contributing text, most recent first. Was a "
                         "hardcoded head(20) in dataframe order, so the feature depended on "
                         "storage order rather than on time.")
    ap.add_argument("--top-keys", type=int, default=0,
                    help="keep only the N keys with the highest standalone validation AUC, "
                         "ranked by the same gate used before building. 0 keeps all. Every "
                         "key's block is otherwise concatenated indiscriminately, and the "
                         "gate already measures them as far apart as 53 to 82 on rel-event.")
    ap.add_argument("--max-columns", default="2",
                    help="per-child column budget, or 'none'. NOT a universal default: the "
                         "base sweep measures max_columns=None at +23.38 on rel-f1's "
                         "validation split against the 2 hardcoded here, while rel-event "
                         "loses 1.36 by the same change. Set it per task.")
    ap.add_argument("--top-children", type=int, default=0,
                    help="keep the N child tables with the strongest single column on "
                         "validation, instead of the first N in dictionary order. WHICH "
                         "children, not how many -- the count has been swept, the identity "
                         "never has. --top-keys does exactly this for link tables and "
                         "measured them 53 to 82 apart on rel-event.")
    ap.add_argument("--categories", type=int, default=0,
                    help="emit a per-category proportion block for each categorical child "
                         "column: the N most frequent values plus an 'other' bucket. 0 is "
                         "off, which is what every standing number was measured under -- "
                         "this runner has never set top_k_categories, so the histogram has "
                         "been dead code in production since it was written.")
    ap.add_argument("--category-share", type=float, default=0.5,
                    help="minimum share of non-null rows the codebook must capture before a "
                         "column gets a histogram. Below it the column is free text in "
                         "disguise (rel-trial's eligibilities.criteria has 247k values and "
                         "its top 4 cover 0.1%%), and the block is K+1 constant columns.")
    ap.add_argument("--numeric-booleans", action="store_true",
                    help="aggregate boolean child columns as numbers, so a boolean history "
                         "yields its rate. They are categorical by default, which gives "
                         "each one a single nunique of 1 or 2 -- and with mode and the "
                         "histogram both off, that has been their entire contribution to "
                         "every number in the table.")
    ap.add_argument("--mode", action="store_true",
                    help="emit the modal value of each categorical child column over its "
                         "all-history prefix. Entity-relative rather than corpus-relative, "
                         "so unlike the histogram it stays meaningful at high cardinality. "
                         "Also never set by this runner before now.")
    ap.add_argument("--timed-links-only", action="store_true",
                    help="use only link tables that carry a timestamp. Required for a "
                         "reportable structural result: an untimed table's degree is "
                         "constant under a cutoff shift, so the temporal control cannot "
                         "see it and passing proves nothing about it.")
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
    max_cols = None if str(args.max_columns).lower() in ("none", "null", "") else int(args.max_columns)
    spec = args.window_days or DEFAULT_WINDOWS.get(args.dataset, "30,365")
    WINDOWS = [pd.Timedelta(days=int(d)) for d in spec.split(",")]
    print(f"{args.dataset}/{args.task}  horizon={horizon}  windows={spec}  "
          f"train={len(train)} test={len(test)}", flush=True)

    # --- base features -------------------------------------------------------------------
    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    all_kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
                for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                if pt == entity and t.time_col]
    kids = all_kids if args.children <= 0 else all_kids[: args.children]
    if args.top_children and len(all_kids) > args.top_children:
        # WHICH child tables, not how many. `--children N` sweeps the count and has been
        # swept; the identity has always been `all_kids[:N]`, which is dictionary order --
        # the same storage-order selection that put `kids[:3]` on rel-trial's three least
        # useful children and picked child text rows by row position instead of by time.
        #
        # `--top-keys` already does this for link tables and measured them 53 to 82 apart
        # on rel-event. There is no reason child tables are more equal than link tables,
        # and no reason dictionary order should find the good ones.
        #
        # Ranked on VALIDATION and univariately, so it costs no fit: a column's own AUC
        # against the validation target needs no model. Distance from chance, because a
        # strongly anti-correlated column is as informative as a correlated one.
        val_rank = task.get_table("val", mask_input_cols=False).df
        truth = val_rank[target].to_numpy()
        shuffled = np.random.default_rng(0).permutation(truth)

        def best_column(block, labels):
            """Largest distance from chance any single column of this block reaches."""
            if len(np.unique(labels)) < 2:
                return 0.0
            best = 0.0
            for col in block.columns:
                values = block[col].to_numpy(dtype=np.float64)
                if not np.isfinite(values).any():
                    continue
                filled = np.nan_to_num(values, nan=float(np.nanmedian(values)),
                                       posinf=0.0, neginf=0.0)
                if len(np.unique(filled)) < 2:
                    continue
                best = max(best, abs(roc_auc_score(labels, filled) * 100 - 50.0))
            return best

        scored = []
        for spec in all_kids:
            n, fk, tc = spec
            table = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS,
                          max_columns=max_cols)
            block = asof_statistics(table, val_rank[key].to_numpy(),
                                    val_rank[tcol].to_numpy())
            # A maximum over k columns grows with k even under pure noise, so ranking
            # children by their best column would rank them by how WIDE they are. The same
            # maximum against a permuted target measures exactly that width, on this
            # block's own column count and missingness -- so the difference is the part
            # that is about signal. Same idea as the permutation control, applied to a
            # selection step rather than to a result.
            signal, null = best_column(block, truth), best_column(block, shuffled)
            scored.append((signal - null, signal, null, len(block.columns), spec))
        scored.sort(key=lambda r: -r[0])
        print("child ranking on validation (best column above its own permuted null):",
              flush=True)
        for margin, signal, null, width, spec in scored:
            print(f"  {spec[0]:<22} {margin:>+6.1f}  (best {signal:.1f}, "
                  f"null {null:.1f}, {width} cols)", flush=True)
        kids = [s[4] for s in scored[: args.top_children]]
    print(f"child tables: using {len(kids)} of {len(all_kids)} available "
          f"{[k[0] for k in kids]}", flush=True)

    def build_base(frame):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS,
                      max_columns=max_cols,
                      top_k_categories=args.categories or None,
                      min_category_share=args.category_share,
                      numeric_booleans=args.numeric_booleans,
                      include_mode=args.mode)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
        out = pd.concat(blocks, axis=1)
        _audit_requested_blocks(out)
        return out

    reported = set()

    def _audit_requested_blocks(frame):
        """Refuse to let a requested feature block be silently empty.

        Depth-2 spent a week "measured at no effect" because `max_columns` was deleting
        every grandchild column before the model saw one: an empty block does not raise,
        it reports +0.00 with sd 0.00 and reads exactly like a clean null. The categorical
        histogram has the same failure mode and its own gate -- `min_category_share` drops
        columns whose codebook captures too little, and on a schema of free-text columns
        that is *all* of them. So count what actually came out, once, and say it.
        """
        for flag, suffix, label in ((args.categories, "__cat0", "category histogram"),
                                    (args.mode, "__mode", "prefix mode")):
            if not flag or label in reported:
                continue
            reported.add(label)
            got = [c for c in frame.columns if c.endswith(suffix)]
            if not got:
                raise SystemExit(
                    f"--{label.split()[0]} was requested but the {label} emitted no columns, "
                    f"so any gap measured here would be an artefact of an empty block. "
                    f"Likely the --category-share gate ({args.category_share}) rejected "
                    f"every categorical column as free text."
                )
            print(f"{label}: {len(got)} columns over "
                  f"{len({c.split('__')[0] for c in got})} child tables", flush=True)

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
    if args.timed_links_only:
        # Control 4 is BLIND to untimed link tables: with no time filter their n_linked
        # does not change when the cutoff is shifted, so it contributes nothing for the
        # control to detect and a pass says nothing about them. Dropping them is what makes
        # a structure-only result defensible rather than merely untested.
        dropped = [s[0] for s in link_specs if not s[4]]
        link_specs = [s for s in link_specs if s[4]]
        if dropped:
            print(f"dropping untimed link tables {dropped}: their structural counts cannot "
                  f"be temporally controlled", flush=True)
    if args.top_keys and len(link_specs) > args.top_keys:
        # Rank on VALIDATION, never on test: this is a selection step like any other, and
        # ranking it on test is the error the calibrated protocol exists to prevent.
        val_rank = task.get_table("val", mask_input_cols=False).df
        scored = []
        for spec in link_specs:
            short, frame, fk, other, ltc = spec
            block = key_target_history(
                frame[[fk, other]], label_entities=train[key].to_numpy(),
                label_values=y, label_times=train[tcol].to_numpy(),
                query_entities=val_rank[key].to_numpy(),
                query_times=val_rank[tcol].to_numpy(), label_horizon=horizon,
                link_times=frame[ltc].to_numpy() if ltc else None)
            rate = block["hist__positive_rate"].to_numpy()
            truth = val_rank[target].to_numpy()
            filled = np.where(np.isnan(rate), y.mean(), rate)
            auc = (roc_auc_score(truth, filled) * 100
                   if len(np.unique(truth)) > 1 else 50.0)
            # Rank by distance from chance: a strongly *anti*-correlated key is as
            # informative as a correlated one, and a rate near 50 is the useless case.
            scored.append((abs(auc - 50.0), auc, spec))
        scored.sort(key=lambda r: -r[0])
        print("key ranking on validation: "
              + ", ".join(f"{s[2][0]} {s[1]:.1f}" for s in scored), flush=True)
        link_specs = [s[2] for s in scored[: args.top_keys]]
        print(f"keeping top {args.top_keys}: {[s[0] for s in link_specs]}", flush=True)

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

    # --- text block (RESEARCH 6f) --------------------------------------------------------
    # Fitted on TRAIN ONLY and applied to val/test. Fitting the vectoriser on all splits
    # would let test vocabulary and IDF weights inform the representation -- a leak that
    # produces a large confident number rather than an error, which is this family's
    # signature failure.
    text_cols, text_models, child_text = [], {}, []
    child_series = None
    if args.text:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline

        ent_text = db.table_dict[entity].df
        merged_tr = train.merge(ent_text, left_on=key, right_on=pk, how="left")

        # Child-table text, aggregated AS OF each row's cutoff. The gate found
        # `eligibilities.criteria` at 63.95 with full coverage -- as strong as the best
        # entity column and not otherwise used. It concatenated without regard to time,
        # so its number was an upper bound; here only rows dated at or before the cutoff
        # contribute. Ignoring that is what inflated rel-event's structural result by 8.
        child_text = []
        for cname, ctbl in db.table_dict.items():
            cfk = next((fk for fk, pt in (ctbl.fkey_col_to_pkey_table or {}).items()
                        if pt == entity), None)
            if cfk is None or not ctbl.time_col:
                continue
            for col in ctbl.df.columns:
                if col in (cfk, ctbl.time_col):
                    continue
                s = ctbl.df[col]
                if not pd.api.types.is_object_dtype(s):
                    continue
                nn = s.dropna().astype(str)
                if len(nn) < 20 or nn.str.len().mean() < 15:
                    continue
                child_text.append((f"{cname}.{col}", cname, cfk, col, ctbl.time_col))
        if child_text:
            print(f"child text columns: {[c[0] for c in child_text]}", flush=True)

        def child_series(frame, spec):
            _, cname, cfk, col, ctc = spec
            src = db.table_dict[cname].df[[cfk, ctc, col]].dropna(subset=[cfk, col])
            q = pd.DataFrame({"_row": np.arange(len(frame)),
                              cfk: frame[key].to_numpy(),
                              "_cut": frame[tcol].to_numpy()})
            j = q.merge(src, on=cfk, how="left")
            j = j[j[ctc].isna() | (j[ctc] <= j["_cut"])]
            # Most RECENT rows, not the first twenty in dataframe order. The cap is a cost
            # control, but taking "whichever pandas happened to store first" made the
            # feature depend on storage order -- and where an entity has more rows than the
            # cap, recency is the only defensible tie-break for an as-of feature.
            j = j.sort_values(ctc, kind="stable", na_position="first")
            agg = (j.dropna(subset=[col]).astype({col: str})
                   .groupby("_row")[col].apply(lambda v: " ".join(v.tail(args.text_rows))))
            out = pd.Series("", index=range(len(frame)), dtype=object)
            out.loc[agg.index] = agg.to_numpy()
            return out

        for spec in child_text:
            merged_tr[spec[0]] = child_series(train, spec).to_numpy()
        entity_texty = [c for c in ent_text.columns
                        if c not in (pk, target)
                        and pd.api.types.is_object_dtype(ent_text[c])
                        and len(ent_text[c].dropna()) >= 20
                        and ent_text[c].dropna().astype(str).str.len().mean() >= 15]
        for col in entity_texty + [c[0] for c in child_text]:
            texts = merged_tr[col].fillna("").astype(str)
            if texts.str.len().sum() == 0:
                continue
            # Components must fit the column's own vocabulary; a fixed 64 killed
            # biospec_retention outright at 11 features.
            try:
                vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000,
                                      ngram_range=(1, 2), strip_accents="unicode")
                n_feat = vec.fit(texts).transform(texts[:1]).shape[1]
                k = int(min(args.text_components, max(2, n_feat - 1)))
                model = make_pipeline(
                    TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000,
                                    ngram_range=(1, 2), strip_accents="unicode"),
                    TruncatedSVD(n_components=k, random_state=0))
                model.fit(texts)
            except Exception as exc:          # noqa: BLE001
                print(f"  text column {col} skipped: {str(exc)[:60]}", flush=True)
                continue
            text_cols.append(col)
            text_models[col] = model
        print(f"text columns embedded: {text_cols or 'none'}", flush=True)

    def text_block(frame):
        if not text_cols:
            return pd.DataFrame(index=range(len(frame)))
        merged = frame.merge(db.table_dict[entity].df, left_on=key, right_on=pk, how="left")
        for spec in child_text:
            if spec[0] in text_cols:
                merged[spec[0]] = child_series(frame, spec).to_numpy()
        blocks = []
        for col in text_cols:
            emb = text_models[col].transform(merged[col].fillna("").astype(str))
            blocks.append(pd.DataFrame(
                emb, columns=[f"txt_{col}_{i}" for i in range(emb.shape[1])]))
        return pd.concat(blocks, axis=1)

    x_tr, x_te = text_block(train), text_block(test)

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
    # Text arms are gated by the label-derived controls only where they include label
    # features. A pure text embedding derives from entity columns, not from any label, so
    # the permutation and temporal controls have nothing to say about it -- and a control
    # that cannot make a feature's value move cannot clear it either.
    ok = {"base": True,
          "+text": True,
          "+text+rate": perm.passed and temporal.passed and temporal_counts.passed,
          "+struct": temporal_struct.passed,
          "+counts": temporal_counts.passed,
          "+rate": perm.passed and temporal.passed and temporal_counts.passed,
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
    def stack(base, block, cols=None, fit=False, tag=""):
        chosen = block if cols is None else block[cols]
        return _numeric(pd.concat([base.reset_index(drop=True),
                                   chosen.reset_index(drop=True)], axis=1),
                        fit=fit, tag=tag or "stack")

    # `+rate` is the label-history block *without* the structural column: n_prior,
    # n_positive, positive_rate. It is the composition that produced rel-trial's 69.36,
    # before n_linked existed. Keeping it separate matters -- folding n_linked into
    # `+history` meant a control failing on the structural column excluded an arm that
    # never contained it, which is a two-variable comparison wearing a verdict's clothing.
    rate_cols = [c for c in t_tr.columns if not c.endswith("n_linked")]
    # Each arm fits its category map on TRAIN and reuses it for val/test, keyed by arm so
    # two arms with different column sets cannot share a stale map.
    arms = {
        "base": (_numeric(b_tr, fit=True, tag="base"), _numeric(b_te, tag="base")),
        "+struct": (stack(b_tr, t_tr, struct_cols, fit=True, tag="struct"),
                    stack(b_te, t_te, struct_cols, tag="struct")),
        **({"+text": (stack(b_tr, x_tr), stack(b_te, x_te)),
            "+text+rate": (stack(pd.concat([b_tr.reset_index(drop=True),
                                            x_tr.reset_index(drop=True)], axis=1),
                                 t_tr, rate_cols),
                           stack(pd.concat([b_te.reset_index(drop=True),
                                            x_te.reset_index(drop=True)], axis=1),
                                 t_te, rate_cols))} if text_cols else {}),
        "+counts": (stack(b_tr, t_tr, count_cols), stack(b_te, t_te, count_cols)),
        "+rate": (stack(b_tr, t_tr, rate_cols), stack(b_te, t_te, rate_cols)),
        "+history": (stack(b_tr, t_tr), stack(b_te, t_te)),
    }
    # An arm whose own controls failed is not offered to validation at all. Selection
    # cannot be allowed to pick a leaking arm and have the protocol launder it.
    arms = {k: v for k, v in arms.items() if ok.get(k, True)}
    # Cheap insurance on the runner that produces the standing table. It drops the target
    # correctly today; `eval_depth2` did not, and scored AUC 100.00 in both arms of a
    # paired comparison whose difference read as a clean +0.00. A regression here would be
    # far more expensive, and a control that only runs when someone suspects something is
    # not a control.
    for name, (X_arm, _) in arms.items():
        assert_no_perfect_feature(X_arm, y, context=f"arm {name!r}")
    widths = ", ".join(f"{k} {v[0].shape[1]}" for k, v in arms.items())
    print(f"feature widths: {widths}; {len(arms['base'][0])} train rows, "
          f"context={args.context}", flush=True)

    def score(X, Xe, rows, seed, truth=None):
        """Fit and score, optionally averaging predictions over several context draws.

        `RESEARCH` item 10. The largest measured weakness on this branch is not bias but
        variance from *which rows land in the context*: rel-event's calibrated replicates
        span 10.8 points and rel-avito's random arm spanned 9.1. Averaging probabilities
        over independent draws attacks that directly.

        It is not `n_estimators`, which varies the model seed at a **fixed** context and was
        measured at nothing on every task -- the ensembling that matters here is over the
        context, which is the thing that actually moves.
        """
        target_y = y_te if truth is None else truth
        draws = max(1, args.resample)
        n = len(X)
        probs = None
        for d in range(draws):
            # Draw d=0 is exactly the unresampled behaviour, so --resample 1 reproduces
            # every earlier number and the comparison stays single-variable.
            if d == 0:
                take = rows
            elif args.stratify_context:
                # Draw each class in proportion to its share of the training set. A uniform
                # draw lets class balance wander between draws, which is noise added to the
                # very quantity resampling exists to average down -- and an accidentally
                # skewed context is not hypothetical here: the graph-context arm once built
                # one at a 0.02 positive rate against a 0.163 base rate.
                r = np.random.default_rng(seed * 1000 + d)
                parts = []
                for cls in np.unique(y):
                    pool = np.flatnonzero(y == cls)
                    want = int(round(len(rows) * len(pool) / n))
                    parts.append(r.choice(pool, size=min(want, len(pool)), replace=False))
                take = np.concatenate(parts)
            else:
                take = np.random.default_rng(seed * 1000 + d).choice(
                    n, size=len(rows), replace=False)
            clf = TabICLClassifier(n_estimators=args.n_estimators, device=args.device,
                                   random_state=seed,
                                   inference_config=NOAMP).fit(X[take], y[take])
            p = clf.predict_proba(Xe)[:, 1]
            probs = p if probs is None else probs + p
        return roc_auc_score(target_y, probs / draws) * 100

    if args.calibrated:
        val = task.get_table("val", mask_input_cols=False).df
        y_va = val[target].to_numpy()
        b_va = build_base(val).reindex(columns=b_tr.columns, fill_value=np.nan)
        t_va = track(val[key].to_numpy(), val[tcol].to_numpy(), y)
        x_va = text_block(val)
        val_arms = {k: v for k, v in {
            "base": _numeric(b_va),
            "+text": stack(b_va, x_va),
            "+text+rate": stack(pd.concat([b_va.reset_index(drop=True),
                                           x_va.reset_index(drop=True)], axis=1),
                                t_va, rate_cols),
            "+struct": stack(b_va, t_va, struct_cols),
            "+counts": stack(b_va, t_va, count_cols),
            "+rate": stack(b_va, t_va, rate_cols),
            "+history": stack(b_va, t_va),
        }.items() if k in arms}
        def cv_score(name, size, seed):
            """Selection criterion from k-fold CV over train, instead of one small split.

            Each fold fits on a subsample of the other folds and scores the held-out one, so
            every training row contributes to the criterion. A 960-row validation split
            cannot resolve a 0.6-point difference; k folds over 12,000 rows can. Test is
            still touched exactly once, after selection -- this changes only *what the
            selection listens to*, not how many times the answer is consulted.
            """
            X, _ = arms[name]
            n = len(X)
            rng = np.random.default_rng(seed)
            if args.cv_time_ordered:
                # Forward chaining: fold k is scored using only rows BEFORE it. Random
                # k-folds break the temporal ordering the benchmark rests on -- a random
                # fold trains on the future to predict the past, which rewarded context
                # resampling with +5.3 of CV while test fell 0.76. A criterion that grows
                # more confident as test degrades is worse than a noisy one.
                order = np.argsort(train[tcol].to_numpy(), kind="stable")
                blocks = np.array_split(order, args.cv_folds + 1)
                folds = blocks[1:]                       # first block is history only
                past = {i: np.concatenate(blocks[:i + 1]) for i in range(len(folds))}
            else:
                order = rng.permutation(n)
                folds = np.array_split(order, args.cv_folds)
                past = None
            scores = []
            for i, f in enumerate(folds):
                if len(np.unique(y[f])) < 2:
                    continue
                rest = past[i] if past is not None else np.setdiff1d(order, f,
                                                                     assume_unique=False)
                if len(rest) < 50:
                    continue
                # Reuse `score`, which already averages over --resample draws. The first
                # version built its own classifier call: it ignored --resample (returning an
                # identical 90.37 for 1 and 3 draws, so the criterion was blind to the very
                # setting it judged) and then threw a torch TypeError the main path never
                # hits. A selection criterion should exercise the scoring path it selects
                # for, not a parallel reimplementation of it.
                take = rng.choice(rest, size=min(size, len(rest)), replace=False)
                scores.append(score(X, X[f], take, seed, truth=y[f]))
            return float(np.mean(scores)) if scores else 0.0

        results = []
        for seed in range(args.seeds):
            best = None
            criterion = f"{args.cv_folds}-fold CV over train" if args.cv_folds \
                else "validation only"
            print(f"\n-- seed {seed}: selection ({criterion}) --", flush=True)
            # Capped at --context, not at the training-set size: rel-avito has 86,619 rows
            # and a grid scaled to that asks for contexts an L40S will not fit, so the run
            # dies rather than reporting a smaller honest number.
            cap = min(len(arms["base"][0]), args.context)
            grid = sorted({max(1000, cap // 4), max(2000, cap // 2), cap})
            for name in arms:
                for size in grid:
                    n = len(arms[name][0])
                    rows = np.random.default_rng(seed).choice(n, size=min(size, n),
                                                              replace=False)
                    v = (cv_score(name, size, seed) if args.cv_folds
                         else score(arms[name][0], val_arms[name], rows, seed, truth=y_va))
                    print(f"  {name:<9} context={size:<6} "
                          f"{'cv' if args.cv_folds else 'val'}={v:.2f}", flush=True)
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
        # (The old warning here said a counts-only choice was uncontrolled. That was true
        # before controls 3 and 4 existed; now every arm is gated by a control on its own
        # columns and an ineligible arm is never offered to validation, so the warning
        # fired on results that *were* controlled. A warning that cries wolf gets ignored.)
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
    means = ", ".join(f"{k} {np.mean(v):.2f}" for k, v in results.items())
    print(f"means: {means}  ({REFERENCE.get(args.dataset, 'see PERFORMANCE.md')})",
          flush=True)
    print("NOTE: the +-0.6 floor applies.", flush=True)


if __name__ == "__main__":
    main()
