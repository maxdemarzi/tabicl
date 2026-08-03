"""Does splitting motifs by edge type, or making them causal, beat the static count?

`eval_relbench_motif.py` established +0.135 AUC on rel-event / user-ignore from three
untyped columns, and the ablation in DESIGN.md attributed ~85% of that to degree alone.
Two refinements are not recoverable from degree, and this script tests both.

**Type splitting.** rel-event has three user-user relations, not one:

  friend     `user_friends`, 213,703 non-null edges, 90.5% of task users, NO timestamp
  coinvite   both users invited to the same event, timestamped by the event
  coattend   both users answered yes/maybe to the same event, timestamped

**Causality.** `user_friends` has no timestamp, so the recorded +0.135 lets a
friendship formed after a prediction time inform that prediction. The co-occurrence
relations are timestamped, so they can be cut strictly before each prediction time.

Dating a co-occurrence edge at the event's `start_time` is conservative: the invitation
was sent before the event, so the edge is credited later than it really formed. That
under-uses information and cannot leak, which is the right direction to err.

Levels
------
  A  relational history only
  B  + static friend motifs        reproduces the recorded result
  C  + typed static motifs         does splitting by relation beat one untyped count
  D  + causal motifs               strictly before the cutoff -- but drops friendship
  E  + causal, windowed, phased    recency and ordering
  F  + static friend AND causal    untyped combination
  G  + static co-occurrence        the control that isolates causality from relation
  H  + typed causal                strictly causal, every relation timestamped
  I  + typed, friend static        H plus friendship, which has no timestamp

**B -> D changes two things at once** -- it adds causality and removes the only
relation covering 90% of task users. G holds the relation fixed and removes only the
cutoff, so B -> G is relation choice and G -> D is causality.

**H is the only strictly causal typed number.** I includes friendship as a static
type, which is worth measuring because dropping a 91.8%-coverage relation is expensive,
but a static type leaks and the result is only as causal as its least causal type.

Usage
-----
    python -m tabicl.scaling.eval_relbench_typed_temporal [levels] [--max-event N]
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
    typed_temporal_motif_features,
)

TASK = "user-ignore"
# Co-occurrence is a clique per event, so a 10,000-invitee event alone would contribute
# 50M pairs and nothing discriminative. The cap is the single most consequential knob
# here, so what it removes is printed rather than left implicit.
DEFAULT_MAX_EVENT = 40
LABELS = {
    "A": "relational only",
    "B": "+ static friend motifs",
    "C": "+ typed static motifs",
    "D": "+ causal motifs",
    "E": "+ causal, windowed, phased",
    "F": "+ static friend AND causal",
    "G": "+ static co-occurrence",
    "H": "+ typed causal (co-occ only)",
    "I": "+ typed, friend static + causal",
}


def numeric(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0]
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def cooccurrence_edges(attendees: pd.DataFrame, statuses, max_event: int):
    """Users who answered the same event alike, timestamped when that event ran.

    A pair that co-occurs on several events is dated by the *earliest* of them: that is
    when the relationship first existed, and any later date would hide it from cutoffs
    that should see it.
    """
    df = attendees[attendees["status"].isin(statuses)]
    df = df[["event", "user_id", "start_time"]].dropna()
    sizes = df.groupby("event")["user_id"].transform("size")
    over = sizes > max_event
    dropped = int(df.loc[over, "event"].nunique())
    df = df[~over]
    if df.empty:
        return np.empty((0, 2)), np.empty(0), dropped

    left = df.rename(columns={"user_id": "u"})
    right = df.rename(columns={"user_id": "v"})[["event", "v"]]
    pairs = left.merge(right, on="event")
    pairs = pairs[pairs["u"] < pairs["v"]]
    if pairs.empty:
        return np.empty((0, 2)), np.empty(0), dropped

    first = pairs.groupby(["u", "v"], as_index=False)["start_time"].min()
    return first[["u", "v"]].to_numpy(), first["start_time"].to_numpy(), dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("levels", nargs="?", default="ABCDEFGHI")
    ap.add_argument("--max-event", type=int, default=DEFAULT_MAX_EVENT)
    ap.add_argument("--n-estimators", type=int, default=4)
    args = ap.parse_args()
    levels = [lv for lv in "ABCDEFGHI" if lv in set(args.levels.upper())]

    db = get_dataset("rel-event").get_db()
    task = get_task("rel-event", TASK)
    key, target = task.entity_col, task.target_col
    train, val = task.get_table("train").df, task.get_table("val").df
    print(f"task={TASK} entity_col={key!r} target={target!r} "
          f"train={len(train)} val={len(val)} pos={train[target].mean():.3f}")
    print(f"cutoffs: {train['timestamp'].nunique()} train + {val['timestamp'].nunique()} val "
          f"distinct, {str(train['timestamp'].min())[:10]} -> {str(val['timestamp'].max())[:10]}")
    print(f"compiled WCOJ backend: {native_available()}")

    users_tbl = db.table_dict["users"].df
    interest = db.table_dict["event_interest"].df
    attendees = db.table_dict["event_attendees"].df

    t0 = time.perf_counter()
    friends = db.table_dict["user_friends"].df[["user", "friend"]].dropna()
    invite_pairs, invite_times, inv_dropped = cooccurrence_edges(
        attendees, ["invited"], args.max_event
    )
    attend_pairs, attend_times, att_dropped = cooccurrence_edges(
        attendees, ["yes", "maybe"], args.max_event
    )
    print(f"edges built in {time.perf_counter()-t0:.0f}s: "
          f"friend={len(friends):,}  coinvite={len(invite_pairs):,} ({inv_dropped:,} events "
          f"over {args.max_event} dropped)  coattend={len(attend_pairs):,} ({att_dropped:,} dropped)")

    # --- one node id space shared by every relation ---------------------------------
    blocks = [
        friends["user"].to_numpy(), friends["friend"].to_numpy(),
        invite_pairs[:, 0], invite_pairs[:, 1],
        attend_pairs[:, 0], attend_pairs[:, 1],
        train[key].to_numpy(), val[key].to_numpy(),
    ]
    codes, uniques = pd.factorize(np.concatenate(blocks))
    lookup = pd.Series(np.arange(len(uniques)), index=uniques)

    cut, sliced = 0, []
    for block in blocks:
        sliced.append(codes[cut:cut + len(block)])
        cut += len(block)
    friend_edges = np.column_stack(sliced[0:2]).astype(np.int64)
    invite_edges = np.column_stack(sliced[2:4]).astype(np.int64)
    attend_edges = np.column_stack(sliced[4:6]).astype(np.int64)

    # Every timestamped edge, as one relation, for the untyped causal levels.
    causal_edges = np.vstack([invite_edges, attend_edges])
    causal_times = np.concatenate([invite_times, attend_times])

    by_type = {"friend": friend_edges, "coinvite": invite_edges, "coattend": attend_edges}
    for name, e in by_type.items():
        reach = np.unique(e) if e.size else np.empty(0, dtype=np.int64)
        seen = lookup.reindex(train[key].to_numpy()).to_numpy()
        print(f"  {name:<9} {len(e):>8,} edges   {np.isin(seen, reach).mean():>6.1%} of task users")

    static_friend = motif_features(friend_edges)
    typed_static = typed_motif_features(by_type)
    # Level G's control: the same edges D sees, with the cutoff removed. G vs D is
    # causality alone; B vs G is relation choice alone. Without it, D - B credits
    # causality for a change that also swapped which relation is being measured.
    static_causal = motif_features(causal_edges)

    def node_ids(entity_df):
        return lookup.reindex(entity_df[key].to_numpy()).to_numpy().astype(np.int64)

    def causal_block(entity_df, windows, n_phases):
        return temporal_motif_features(
            causal_edges,
            causal_times,
            nodes=node_ids(entity_df),
            cutoffs=entity_df["timestamp"].to_numpy(),
            windows=windows,
            n_phases=n_phases,
        )

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
        if level in ("B", "F"):
            joined = static_friend.reindex(ids)
            for col in static_friend.columns:
                feats[f"friend__{col}"] = joined[col].to_numpy()
        if level == "C":
            joined = typed_static.reindex(ids)
            for col in typed_static.columns:
                feats[col] = joined[col].to_numpy()
        if level == "G":
            joined = static_causal.reindex(ids)
            for col in static_causal.columns:
                feats[f"cooc__{col}"] = joined[col].to_numpy()
        if level in ("H", "I"):
            # H is strictly causal: every relation in it carries a timestamp. I adds
            # friendship as a *static* type, which leaks -- the whole point of keeping
            # them separate is that only H's number is a causal one.
            if level == "H":
                edge_types = {"coinvite": invite_edges, "coattend": attend_edges}
                time_types = {"coinvite": invite_times, "coattend": attend_times}
            else:
                edge_types = dict(by_type)
                time_types = {"friend": None, "coinvite": invite_times,
                              "coattend": attend_times}
            typed_causal = typed_temporal_motif_features(
                edge_types,
                time_types,
                nodes=ids,
                cutoffs=entity_df["timestamp"].to_numpy(),
            )
            for col in typed_causal.columns:
                feats[f"tc__{col}"] = typed_causal[col].to_numpy()
        if level in ("D", "E", "F"):
            if level == "D":
                windows, phases = {"all": None}, 1
            else:
                windows, phases = {"all": None, "d30": pd.Timedelta("30D")}, 2
            causal = causal_block(entity_df, windows, phases)
            for col in causal.columns:
                feats[f"causal__{col}"] = causal[col].to_numpy()
        return feats

    y_tr, y_va = train[target].to_numpy(), val[target].to_numpy()
    print()
    for level in levels:
        t0 = time.perf_counter()
        f_tr, f_va = build(train, level), build(val, level)
        f_va = f_va.reindex(columns=f_tr.columns, fill_value=np.nan)
        X_tr, X_va = numeric(f_tr), numeric(f_va)

        clf = TabICLClassifier(
            n_estimators=args.n_estimators, device="cpu", random_state=0
        ).fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        gbdt = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
        auc_g = roc_auc_score(y_va, gbdt.predict_proba(X_va)[:, 1])
        print(f"  {level} {LABELS[level]:<28} features={X_tr.shape[1]:>4}  "
              f"TabICL AUC={auc:.4f}   GBDT AUC={auc_g:.4f}   ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
