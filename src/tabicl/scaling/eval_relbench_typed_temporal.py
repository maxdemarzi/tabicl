"""Does splitting motifs by edge type, or making them causal, beat the static count?

`eval_relbench_motif.py` established +0.135 AUC on rel-event / user-ignore from three
untyped columns, and the ablation in DESIGN.md then attributed ~85% of that to degree
alone. Two things follow from the literature, and this script tests both.

**Type splitting.** Most of what an untyped triangle count carries is degree in
disguise, which is the expected result: motif counts are partly determined by the
degree sequence. A *typed* count is not recoverable that way, which is the argument
Lichtenwalter & Chawla make for vertex collocation profiles. rel-event has two
user-user relations to split on -- declared friendship, and co-interest in the same
event -- so the split is available without leaving the dataset.

**Causality.** The recorded +0.135 is an upper bound, because `user_friends` carries
no timestamp and a friendship formed after a prediction time is still visible to it.
`event_interest` *is* timestamped, so a co-interest graph can be built strictly before
each prediction time. Levels D and E use only timestamped edges, which makes them the
first strictly causal graph features here -- and the honest comparison for B.

Levels
------
  A  relational history only
  B  + static friendship motifs           (reproduces the recorded result; leaky)
  C  + typed static motifs                (friend / co-interest, still leaky)
  D  + causal co-interest motifs          (strictly before the cutoff)
  E  + causal, windowed, phase-split      (recency and ordering)

Usage
-----
    python -m tabicl.scaling.eval_relbench_typed_temporal [levels] [--max-event N]

``levels`` is a subset of ``ABCDE``; default is all of them.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import (
    Table,
    flatten_relational,
    motif_features,
    native_available,
    temporal_motif_features,
    typed_motif_features,
)

TASK = "user-ignore"
# Co-interest is a clique per event, so one popular event dominates the edge list and
# contributes nothing discriminative. Events above this many interested users are
# dropped; the count of what that removed is printed rather than left implicit.
DEFAULT_MAX_EVENT = 200


def numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def event_column(db, table: str) -> str:
    """Find the column of ``table`` that points at the events table.

    Read from the schema rather than assumed, because guessing a foreign key name is
    exactly the kind of thing that silently produces a wrong graph instead of an error.
    """
    fkeys = getattr(db.table_dict[table], "fkey_col_to_pkey_table", {}) or {}
    targets = [col for col, points_to in fkeys.items() if points_to != "users"]
    if len(targets) != 1:
        raise RuntimeError(
            f"cannot identify the event column of {table!r}: foreign keys are {fkeys}"
        )
    return targets[0]


def cointerest_edges(interest: pd.DataFrame, user_col: str, event_col: str, time_col: str,
                     max_event: int):
    """Users who expressed interest in the same event, timestamped when the pair closed.

    An edge exists from the moment *both* endpoints have shown interest, so the pair
    carries the later of the two timestamps. Dating it any earlier would leak.
    """
    df = interest[[user_col, event_col, time_col]].dropna()
    sizes = df.groupby(event_col)[user_col].transform("size")
    dropped = int((sizes > max_event).groupby(df[event_col]).first().sum())
    df = df[sizes <= max_event]

    left = df.rename(columns={user_col: "u", time_col: "tu"})
    right = df.rename(columns={user_col: "v", time_col: "tv"})
    pairs = left.merge(right, on=event_col)
    pairs = pairs[pairs["u"] < pairs["v"]]
    if pairs.empty:
        return np.empty((0, 2), dtype=object), np.empty(0), dropped

    # Same pair can co-occur on several events; keep the earliest such moment.
    when = pairs[["tu", "tv"]].max(axis=1)
    pairs = pairs.assign(when=when).groupby(["u", "v"], as_index=False)["when"].min()
    return pairs[["u", "v"]].to_numpy(), pairs["when"].to_numpy(), dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("levels", nargs="?", default="ABCDE")
    ap.add_argument("--max-event", type=int, default=DEFAULT_MAX_EVENT)
    args = ap.parse_args()
    levels = set(args.levels.upper())

    db = get_dataset("rel-event", download=True).get_db()
    task = get_task("rel-event", TASK, download=True)
    key, target = task.entity_col, task.target_col
    train, val = task.get_table("train").df, task.get_table("val").df
    print(f"task={TASK} entity_col={key!r} target={target!r} "
          f"train={len(train)} val={len(val)} pos={train[target].mean():.3f}")
    print(f"compiled WCOJ backend: {native_available()}")

    users_tbl = db.table_dict["users"].df
    interest = db.table_dict["event_interest"].df
    ev_col = event_column(db, "event_interest")

    # --- one node id space shared by every relation --------------------------------
    friends = db.table_dict["user_friends"].df[["user", "friend"]].dropna()
    t0 = time.perf_counter()
    co_pairs, co_times, dropped = cointerest_edges(
        interest, "user", ev_col, "timestamp", args.max_event
    )
    print(f"co-interest: {len(co_pairs):,} edges in {time.perf_counter()-t0:.1f}s "
          f"({dropped:,} events over {args.max_event} interested users dropped)")

    universe = np.concatenate([
        friends["user"].to_numpy(), friends["friend"].to_numpy(),
        co_pairs[:, 0] if len(co_pairs) else np.empty(0),
        co_pairs[:, 1] if len(co_pairs) else np.empty(0),
        train[key].to_numpy(), val[key].to_numpy(),
    ])
    codes, uniques = pd.factorize(universe)
    lookup = pd.Series(np.arange(len(uniques)), index=uniques)

    n_fr = len(friends)
    friend_edges = np.column_stack([codes[:n_fr], codes[n_fr:2 * n_fr]]).astype(np.int64)
    start = 2 * n_fr
    co_edges = np.column_stack(
        [codes[start:start + len(co_pairs)], codes[start + len(co_pairs):start + 2 * len(co_pairs)]]
    ).astype(np.int64)
    print(f"graph: {len(friend_edges):,} friendship + {len(co_edges):,} co-interest edges "
          f"over {len(uniques):,} users")

    # --- feature blocks -------------------------------------------------------------
    static_friend = motif_features(friend_edges)
    typed_static = typed_motif_features({"friend": friend_edges, "cointerest": co_edges})

    def node_ids(entity_df):
        return lookup.reindex(entity_df[key].to_numpy()).to_numpy()

    def build(entity_df, level):
        base = entity_df.merge(users_tbl, left_on=key, right_on="user_id", how="left")
        base = base[[c for c in base.columns if c != target]]
        feats = flatten_relational(
            base.drop(columns=["index"], errors="ignore"),
            key,
            [Table(interest, "user", "int", time_column="timestamp")],
            cutoff_column="timestamp",
        )
        if level == "A":
            return feats

        ids = node_ids(entity_df)
        if level == "B":
            joined = static_friend.reindex(ids)
            for col in static_friend.columns:
                feats[f"friend__{col}"] = joined[col].to_numpy()
        elif level == "C":
            joined = typed_static.reindex(ids)
            for col in typed_static.columns:
                feats[col] = joined[col].to_numpy()
        else:
            windows = {"all": None} if level == "D" else {"all": None, "d30": pd.Timedelta("30D")}
            causal = temporal_motif_features(
                co_edges,
                co_times,
                nodes=np.nan_to_num(ids, nan=-1).astype(np.int64),
                cutoffs=entity_df["timestamp"].to_numpy(),
                windows=windows,
                n_phases=1 if level == "D" else 2,
            )
            for col in causal.columns:
                feats[f"co__{col}"] = causal[col].to_numpy()
        return feats

    y_tr, y_va = train[target].to_numpy(), val[target].to_numpy()
    labels = {
        "A": "relational only",
        "B": "+ static friend motifs",
        "C": "+ typed static motifs",
        "D": "+ causal co-interest",
        "E": "+ causal, windowed, phased",
    }
    for level in "ABCDE":
        if level not in levels:
            continue
        t0 = time.perf_counter()
        f_tr, f_va = build(train, level), build(val, level)
        f_va = f_va.reindex(columns=f_tr.columns, fill_value=np.nan)
        X_tr, X_va = numeric(f_tr), numeric(f_va)

        clf = TabICLClassifier(n_estimators=4, device="cpu", random_state=0).fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        gbdt = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
        auc_g = roc_auc_score(y_va, gbdt.predict_proba(X_va)[:, 1])
        print(f"  {level} {labels[level]:<28} features={X_tr.shape[1]:>4}  "
              f"TabICL AUC={auc:.4f}   GBDT AUC={auc_g:.4f}   ({time.perf_counter()-t0:.0f}s)")

    ids = node_ids(train)
    in_friend = static_friend.reindex(ids)["degree"].notna().mean()
    print(f"\ntask users in the friendship graph: {in_friend:.1%}")


if __name__ == "__main__":
    main()
