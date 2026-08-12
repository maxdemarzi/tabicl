"""Do triangles carry signal a model cannot already get from degree?

A SCREEN, not a result. It costs CPU minutes and no TabICL fit, and it exists so that a
round is spent only where the answer might be yes. `DESIGN.md` puts the join machinery at
**+0.016 AUC** on rel-event (degree versus degree+triangles) and **+0.021** with typing --
an order of magnitude below what free features and relation choice deliver there. Motif
counts are partly *determined* by the degree sequence (Bhat et al. 2014 on conservation
laws; a k-star count is exactly `C(d, k)`), so "the graph has triangles" predicts nothing.
The question is always whether they survive degree.

**THE BAR IS PRE-REGISTERED, AND IT IS NOT ZERO.** rel-event's +0.016 was measured and then
judged not worth building on. So a new task earns a round only by beating the task we
already declined:

    degree+triangles+clustering must beat degree alone by MORE THAN +0.016 AUC.

Anything at or below that reproduces a result we have, on a graph we would have to pay to
build. Written here before the first run so the threshold cannot drift to meet the number.

**Why rel-arxiv / paper-citation is the first target.** Its `citations` table is
self-referential and **timestamped** (`Paper_ID` -> `References_Paper_ID`, dated by
`Submission_Date`), which none of the standing twelve provides -- rel-event's best relation,
`user_friends`, carries no timestamp at all, so the configuration that used it is only as
causal as its least causal relation. `DESIGN.md` also records the windows-and-phases result
as **-0.014 on rel-event** with an explicit reason and an explicit request:

    rel-event's prediction window is five months against event timestamps that only reach
    13.9% coverage by the midpoint cutoff, so most rows see too little history for a phase
    split to mean anything. A denser, longer-running temporal graph is where it would get a
    fair test.

Decades of submissions with every citation dated is that graph. The phase split is reported
as a secondary readout for exactly that reason.

**And the honest weakness, stated first so a positive result is read correctly.** "Will this
paper be cited in the next six months" is preferential attachment in its purest form: the
label IS popularity, and degree should dominate. That makes this a hard case for the density
hypothesis rather than a favourable one -- which is the point. Triangles surviving degree
*here* would be strong evidence; failing to is weak evidence against, because the label was
never the density question. `AuthorCategoryTask` (co-author community predicting research
area) is the better-matched label and needs multiclass support this harness lacks.

Usage
-----
    python -m tabicl.scaling.screen_motif_signal rel-arxiv paper-citation
    python -m tabicl.scaling.screen_motif_signal rel-arxiv paper-citation --rows 50000
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# The bar this screen tests against: rel-event's measured degree -> degree+triangles gain,
# which was judged not worth building on. Beating it is the whole point.
REL_EVENT_TRIANGLE_GAIN = 0.016


def load(dataset: str, task: str):
    """Dataset and task, printing what actually exists when a name is wrong.

    Guessing a registry name and reporting the guess as a failure has cost this project a
    pod cycle before (the TabFM loader, RESEARCH.md). rel-arxiv is newer than most of this
    package, so the task string is exactly the kind of thing to be wrong about.
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task, get_task_names

    try:
        db = get_dataset(dataset, download=True).get_db()
    except Exception as exc:
        # Guarded: an error path that itself raises replaces a useful message with a
        # NameError about the reporting code, which is how a wrong dataset name turns into
        # a bug hunt in the wrong file.
        try:
            from relbench.datasets import get_dataset_names
            names = sorted(get_dataset_names())
        except Exception:
            names = "(could not list datasets)"
        raise SystemExit(f"could not load dataset {dataset!r}: {exc}\navailable: {names}")
    try:
        t = get_task(dataset, task, download=True)
    except Exception as exc:
        raise SystemExit(f"could not load task {dataset}/{task!r}: {exc}\n"
                         f"available for {dataset}: {sorted(get_task_names(dataset))}")
    return db, t


def edge_table(db, verbose: bool = True):
    """The self-referential, timestamped edge list this screen needs.

    Found by structure rather than by name: a table whose two foreign keys point at the
    SAME parent is an edge list over that parent, whatever it is called. A hard-coded
    'citations' would work on one dataset and silently do nothing on the next.
    """
    for name, tbl in db.table_dict.items():
        parents = list(tbl.fkey_col_to_pkey_table.items())
        if len(parents) == 2 and parents[0][1] == parents[1][1] and tbl.time_col:
            src, dst = parents[0][0], parents[1][0]
            if verbose:
                print(f"edge table {name!r}: {src} -> {dst} into {parents[0][1]!r}, "
                      f"{len(tbl.df):,} edges dated by {tbl.time_col!r}", flush=True)
            return tbl, src, dst
    raise SystemExit(
        "no self-referential timestamped table found; this screen needs an edge list over "
        "one entity (e.g. citations: Paper_ID -> References_Paper_ID). Tables seen: "
        + ", ".join(f"{n}({list(t.fkey_col_to_pkey_table)})" for n, t in db.table_dict.items()))


def features(edges, times, nodes, cutoffs, n_phases: int = 1):
    from tabicl.scaling import temporal_motif_features
    return temporal_motif_features(edges, times, nodes, cutoffs, n_phases=n_phases)


def auc_with(cols, tr_X, tr_y, te_X, te_y, seed: int) -> float:
    """One GBDT, fixed settings, scored on held-out rows.

    A GBDT rather than TabICL on purpose: the screen asks whether the INFORMATION is
    present, not what our model does with it, and a screen that costs a GPU hour is one
    nobody runs before committing a round.
    """
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    m.fit(tr_X[cols], tr_y)
    return roc_auc_score(te_y, m.predict_proba(te_X[cols])[:, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("task")
    ap.add_argument("--rows", type=int, default=100_000,
                    help="cap on task rows per split. Edges are NEVER subsampled -- the "
                         "graph is the thing being measured, and a thinned graph would "
                         "understate triangles while leaving degree nearly intact, which "
                         "biases this screen toward its own negative result.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--phases", type=int, default=2,
                    help="secondary readout: the ordered census that lost 0.014 on "
                         "rel-event for want of history. 1 disables it.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    db, task = load(args.dataset, args.task)
    tbl, src, dst = edge_table(db)

    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    tcol, ycol = task.time_col, task.target_col
    ecol = getattr(task, "entity_col", None) or next(
        c for c in train.columns if c not in (tcol, ycol))
    print(f"task {args.dataset}/{args.task}: entity {ecol!r}, target {ycol!r}, "
          f"{len(train):,} train / {len(val):,} val rows, "
          f"base rate {train[ycol].mean():.3f}", flush=True)

    if len(train) > args.rows:
        train = train.sample(args.rows, random_state=0).reset_index(drop=True)
    if len(val) > args.rows:
        val = val.sample(args.rows, random_state=0).reset_index(drop=True)

    e = tbl.df[[src, dst]].to_numpy()
    et = tbl.df[tbl.time_col].to_numpy()
    ok = ~(pd.isna(tbl.df[src]) | pd.isna(tbl.df[dst]) | pd.isna(tbl.df[tbl.time_col]))
    e, et = e[ok.to_numpy()].astype(np.int64), et[ok.to_numpy()]

    # THE COST MODEL, PRINTED BEFORE IT IS PAID. `temporal_motif_features` runs one triangle
    # census per distinct cutoff over every edge before it, so the bill is
    # (distinct cutoffs x edges), not (rows). RelBench task tables share a handful of
    # prediction timestamps, which is what makes this affordable -- but on a citation graph
    # of millions of edges it is worth seeing the multiplier before waiting on it rather
    # than after.
    cuts = len(np.unique(np.concatenate([train[tcol].to_numpy(), val[tcol].to_numpy()])))
    print(f"census cost: {cuts} distinct cutoffs x up to {len(e):,} edges "
          f"({args.phases} phase(s))", flush=True)

    frames = {}
    for split, df in (("train", train), ("val", val)):
        s = time.perf_counter()
        f = features(e, et, df[ecol].to_numpy(), df[tcol].to_numpy(), n_phases=args.phases)
        frames[split] = f.reset_index(drop=True)
        print(f"  {split}: motif census in {time.perf_counter() - s:.0f}s, "
              f"{f.shape[1]} columns", flush=True)

    tr, va = frames["train"], frames["val"]
    tr_y, va_y = train[ycol].to_numpy(), val[ycol].to_numpy()

    # `temporal_motif_features` names columns `{window}__degree`, `{window}__triangles`,
    # `{window}__clustering` and `{window}__tri__p0_p0_p1` -- window-prefixed, so matching
    # on a leading "degree" finds nothing and the screen would compare an empty feature set
    # against another empty one and call it a null. Matched on the suffix, and asserted.
    deg = [c for c in tr.columns if c.endswith("__degree")]
    tri = [c for c in tr.columns if c.endswith(("__triangles", "__clustering"))]
    phase = [c for c in tr.columns if "__tri__" in c]
    if not deg or not tri:
        raise SystemExit(f"expected degree and triangle columns; got {list(tr.columns)}")
    tri_only = [c for c in tr.columns if c.endswith("__triangles")]

    # ISOLATED, because a positive screen on a task whose label is popularity is exactly
    # what a degree-driven artefact looks like. Reported alongside so a reader can see
    # how much of the ceiling degree already reaches on its own -- and because a graph where
    # almost no row has a triangle yet cannot answer the question either way.
    print(f"\ncoverage: {100 * (tr[deg].sum(axis=1) > 0).mean():.1f}% of train rows have "
          f"any edge before their cutoff; "
          f"{100 * (tr[tri_only].sum(axis=1) > 0).mean():.1f}% have a triangle", flush=True)

    rows = []
    for seed in range(args.seeds):
        a = auc_with(deg, tr, tr_y, va, va_y, seed)
        b = auc_with(deg + tri, tr, tr_y, va, va_y, seed)
        c = auc_with(deg + tri + phase, tr, tr_y, va, va_y, seed) if phase else np.nan
        rows.append((a, b, c))
        print(f"seed {seed}: degree {a:.4f}  +triangles {b:.4f}  "
              f"({b - a:+.4f})" + (f"  +phases {c:.4f} ({c - b:+.4f})" if phase else ""),
              flush=True)

    r = np.array(rows, dtype=float)
    gain = r[:, 1] - r[:, 0]
    sd = gain.std(ddof=1) if len(gain) > 1 else 0.0
    print(f"\ndegree alone            {r[:, 0].mean():.4f}")
    print(f"+ triangles, clustering {r[:, 1].mean():.4f}   gain {gain.mean():+.4f} "
          f"(sd {sd:.4f}, {(gain > 0).sum()}/{len(gain)} positive)")
    if phase:
        pg = r[:, 2] - r[:, 1]
        print(f"+ ordered phase census  {r[:, 2].mean():.4f}   gain {pg.mean():+.4f}"
              f"   [rel-event: -0.014, for want of history]")

    # THE PRE-REGISTERED CALL, made against the threshold in the docstring and not against
    # whatever this run produced.
    print(f"\nbar: beat rel-event's {REL_EVENT_TRIANGLE_GAIN:+.3f}, the gain already "
          f"measured and already declined.")
    if gain.mean() > REL_EVENT_TRIANGLE_GAIN:
        print(f"VERDICT: PASS ({gain.mean():+.4f}). Triangles carry signal beyond degree on "
              f"a label that is itself popularity, which is the hard case. A round is "
              f"justified; the next question is whether TabICL can use what a GBDT found.")
    else:
        print(f"VERDICT: FAIL ({gain.mean():+.4f}). At or below what rel-event already gave, "
              f"so a round would re-measure a known result on a new graph. Note this is weak "
              f"evidence AGAINST the density hypothesis, not strong: the label here is "
              f"popularity, so the question it answers is not the one the machinery is for.")
    print(f"\ntotal {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
