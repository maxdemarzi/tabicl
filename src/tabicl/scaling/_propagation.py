"""Neighbours' labels as features -- the direct attack on the i.i.d. limitation.

`RESEARCH.md` item 6b. Flattening can aggregate a neighbour's *features* but never its
*label*, and that ceiling is what separates this pipeline from the relational systems it
is measured against. Label propagation removes it: a query's feature vector gains "what
fraction of my labelled neighbours were positive".

**It is already measured on rel-event, and it is strong.** Scoring each query by nothing
but that fraction -- no model at all, one pass over the edge list -- reaches ROC-AUC
**73.89 on validation and 74.18 on test**, against a full calibrated pipeline at 78.11 on
that task. Unlike graph-neighbour *context* (item 8), which measured +3 to +5 and was
still rejected by its own selection rule, this is an ordinary feature: validation picks it
the same way it picks `max_columns`, so it can actually reach a reported number.

Why this module is written defensively
--------------------------------------
This is the highest-risk family in the package. A leak here does not raise; it returns a
large, confident, entirely fake number, and several conclusions in `DESIGN.md` were wrong
for exactly that reason. Three failure modes are designed out rather than tested for
afterwards:

**A row reading its own label.** Self-loops are dropped and a node is never its own
neighbour, so a training row's own outcome cannot re-enter through its own neighbourhood.
Use `permutation_control` from `_leakage.py` to confirm it on real data.

**A label read before it exists.** A neighbour's outcome is not knowable at the neighbour's
own prediction time -- it is realised over the following window. Passing `label_horizon`
(RelBench's ``task.timedelta``) makes a neighbour's label usable only once that window has
closed at or before the query's cutoff. Without it, "the friend who churned last week"
silently includes friends whose churn is still being decided.

**"No neighbours" encoded as "no positive neighbours".** An isolated query gets NaN, not
0.0. Encoding absence as a zero rate puts isolated rows at the bottom of the ranking as
though they were confidently negative, which is a bias, not a measurement.

Scope: one hop
--------------
Deliberately, and on evidence rather than convenience. On rel-event a single hop from the
2,013 validation queries already reaches 7,111 of the 8,517 eligible training users -- so
two hops is the whole connected component, every query gets the same neighbourhood, and
the feature goes constant. `hops_to_nearest_positive` and the `MIN_PLUS` variants in
`RESEARCH.md` need a graph that does not saturate; they are item 9's problem, together
with path-time-ordering. On a graph with no edge timestamps, "labels usable at the cutoff
among one-hop neighbours" is already the causally correct definition.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["neighbour_label_features", "key_target_history", "entity_label_history"]


def entity_label_history(
    label_entities: np.ndarray,
    label_values: np.ndarray,
    label_times: np.ndarray,
    query_entities: np.ndarray,
    query_times: np.ndarray,
    label_horizon: pd.Timedelta,
    prefix: str = "self__",
) -> pd.DataFrame:
    """This entity's OWN earlier outcomes: "how has this driver been finishing?"

    The sibling of `key_target_history`, which deliberately excludes the entity itself and
    reaches other entities through a shared foreign key. This is the piece that leaves:
    the task table is a timestamped table keyed by entity, and until now the pipeline read
    only its key, its cutoff and its label, never the labels of that entity's earlier rows.

    Gate measured before this was built, horizon-correct, standalone test AUC against the
    full pipeline: rel-f1/driver-top3 **84.66** vs 81.98, rel-f1/driver-dnf **74.27** vs
    69.66, rel-event/user-ignore 81.72 vs 80.98, rel-event/user-repeat 67.17 vs 77.89,
    rel-avito 57-59 vs 65-66. It is worth most exactly where entities recur most, and
    `coverage` below reports that per call rather than leaving it to be assumed.

    **rel-trial cannot use this and needs no experiment**: `entities == rows` there, every
    study has one outcome, and coverage is 0.0%.

    Parameters
    ----------
    label_entities, label_values, label_times : np.ndarray
        The outcome events the caller is entitled to know -- normally the **training**
        rows. Pass the fitting pool and nothing else; this function does not check which
        split an event came from, so passing test outcomes would leak and no assertion
        here would catch it.

    query_entities, query_times : np.ndarray
        One per query row, with its cutoff. Returned rows are in this order.

    label_horizon : pd.Timedelta
        RelBench's ``task.timedelta``. **Required, and required to be positive**, which is
        what makes self-exclusion structural rather than asserted -- see Notes.

    Returns
    -------
    pd.DataFrame
        ``{prefix}n_prior``, ``{prefix}positive_rate``, ``{prefix}last``,
        ``{prefix}days_since``. The rate is NaN where no earlier outcome has resolved,
        never 0.0, so "no track record" stays distinguishable from "a uniformly bad one".

    Notes
    -----
    **A row cannot see its own label, by construction rather than by filtering.** An event
    becomes readable at ``label_time + label_horizon`` and a query reads events with
    ``ready <= cutoff``. The query's own event has ``ready = cutoff + label_horizon``,
    which exceeds ``cutoff`` for any positive horizon. There is no self-exclusion step to
    get wrong, which is the point: `key_target_history` has one, and it aggregates the
    entity's events with ``min``/``first``, so on an entity holding several outcomes it
    subtracts the earliest rather than the row's own. That is conservative rather than
    leaky, but only because the horizon already did the real work.

    The horizon is not decoration. RelBench labels answer "does X happen within
    `task.timedelta` of this cutoff", so an outcome recorded at *t* is not known until
    *t* + horizon. Ignoring it overstated this feature by 2.26 AUC on driver-top3. The
    reason the error was not larger is that RelBench spaces an entity's task rows exactly
    one horizon apart -- which is a property of the benchmark, not a licence to omit it.
    """
    horizon = pd.Timedelta(label_horizon)
    if horizon <= pd.Timedelta(0):
        raise ValueError(
            f"label_horizon must be positive, got {horizon!r}. A zero or negative horizon "
            "lets a row read its own outcome: self-exclusion here is structural, and this "
            "is the condition it rests on."
        )

    n_query = len(query_entities)
    cols = [f"{prefix}n_prior", f"{prefix}positive_rate", f"{prefix}last",
            f"{prefix}days_since"]
    out = pd.DataFrame({cols[0]: np.zeros(n_query, dtype=np.int64),
                        cols[1]: np.full(n_query, np.nan),
                        cols[2]: np.full(n_query, np.nan),
                        cols[3]: np.full(n_query, np.nan)})
    if n_query == 0 or len(label_entities) == 0:
        return out

    events = pd.DataFrame({
        "entity": np.asarray(label_entities),
        "ready": pd.Series(np.asarray(label_times)) + horizon,
        "y": np.asarray(label_values, dtype=np.float64),
    }).dropna(subset=["entity", "ready"]).sort_values("ready", kind="stable")
    if not len(events):
        return out
    events["cum_y"] = events.groupby("entity")["y"].cumsum()
    events["cum_n"] = events.groupby("entity").cumcount() + 1

    queries = pd.DataFrame({"row": np.arange(n_query),
                            "entity": np.asarray(query_entities),
                            "cutoff": np.asarray(query_times)})
    ok = queries["entity"].notna() & queries["cutoff"].notna()
    matched = pd.merge_asof(
        queries[ok].sort_values("cutoff", kind="stable"),
        events[["entity", "ready", "y", "cum_y", "cum_n"]],
        left_on="cutoff", right_on="ready", by="entity",
        direction="backward", allow_exact_matches=True,
    ).dropna(subset=["cum_n"])
    if not len(matched):
        return out

    rows = matched["row"].to_numpy()
    n_prior = matched["cum_n"].to_numpy()
    out.loc[rows, cols[0]] = n_prior.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out.loc[rows, cols[1]] = matched["cum_y"].to_numpy() / n_prior
    out.loc[rows, cols[2]] = matched["y"].to_numpy()
    # Time since that outcome became READABLE, not since it was recorded. The second would
    # be a constant offset from the first on this benchmark and invites reading it as
    # "time since the event", which it is not.
    out.loc[rows, cols[3]] = (
        (matched["cutoff"].to_numpy() - matched["ready"].to_numpy())
        / np.timedelta64(1, "D")
    )
    out[cols[0]] = out[cols[0]].astype(np.int64)
    return out


def key_target_history(
    links: pd.DataFrame,
    label_entities: np.ndarray,
    label_values: np.ndarray,
    label_times: np.ndarray,
    query_entities: np.ndarray,
    query_times: np.ndarray,
    label_horizon: Optional[pd.Timedelta] = None,
    link_times: Optional[np.ndarray] = None,
    prefix: str = "hist__",
) -> pd.DataFrame:
    """Track record among earlier rows sharing a key: "how did this sponsor's trials go?"

    The same idea as `neighbour_label_features`, reached through the **schema** instead of
    a graph. Two entities are related because they share a foreign key -- a sponsor, a
    condition, a facility -- and the feature is the outcome history of the others, as of
    this row's cutoff. A generic flattener cannot produce this: it aggregates a related
    row's *columns*, never its *outcome*, and the outcome is what carries the base rate.

    Gate measured on rel-trial before this was built: condition 60.4, sponsor 61.4,
    facility 60.6 standalone test AUC at 76-88% coverage, against a full pipeline at 66.50.

    Parameters
    ----------
    links : pd.DataFrame
        Two columns, ``[entity, key]``, one row per membership. Duplicates are dropped. An
        entity may hold many keys and a key many entities.

    label_entities, label_values, label_times : np.ndarray
        The **training** outcome events: which entity, the outcome, and the prediction time
        it was recorded against. Never pass validation or test outcomes.

    query_entities, query_times : np.ndarray
        One per query row, with its cutoff.

    label_horizon : pd.Timedelta, optional
        How long an outcome takes to resolve after its own prediction time -- RelBench's
        ``task.timedelta``, which is **365 days** on rel-trial. An outcome becomes usable at
        ``label_time + label_horizon``. Omitting it on that task lets a row read a year of
        outcomes that had not happened yet.

    prefix : str, default="hist__"
        Column name prefix.

    Returns
    -------
    pd.DataFrame
        One row per query, in input order: ``{prefix}n_prior``, ``{prefix}n_positive``,
        ``{prefix}positive_rate``, ``{prefix}n_linked``. The rate is NaN where no prior
        outcome is visible, never 0.0, so "no track record" is distinguishable from "a
        uniformly bad one".

        ``n_linked`` counts entities sharing a key **without consulting labels at all**.
        Keep it separate in any comparison: on rel-event most of what looked like a
        label-history gain turned out to be this, and a run that mixes the two cannot tell
        the difference.

    Notes
    -----
    An entity holding several of the query's keys is counted once per shared key, so the
    rate is a membership-weighted average rather than a distinct-entity one.

    Self-exclusion is done by **subtracting** the row's own contribution, not by dropping
    rows whose most recent event happens to be its own. Dropping would discard that key's
    entire accumulated history rather than one observation. On rel-trial the 365-day
    horizon already pushes a row's own outcome past its own cutoff, so this never fires
    there -- which is exactly why it must not be left to that coincidence.

    **That subtraction assumes one outcome per entity**, because ``own`` aggregates the
    entity's events with ``min``/``first``. True on rel-trial, where a study has a single
    outcome; false wherever entities recur, as users do on rel-event. Where an entity holds
    several outcomes it subtracts the *earliest*, not the row's own.

    With a horizon this is harmless and never fires: the row's own event resolves after its
    own cutoff, so ``inside`` is False. Every call site in `eval_track_record` passes
    ``label_horizon=task.timedelta``, so no reported number is affected. **Under
    ``--no-horizon`` on a recurring-entity task it is a leak** -- the wrong observation is
    removed and the row's own label stays in the sum. That flag is already documented as
    wrong and exists only to price its own wrongness, but the price is higher than it
    advertises: not just a year of unresolved outcomes, but the row's own. For the
    entity's *own* track record use `entity_label_history`, where the horizon makes
    self-exclusion structural and there is no subtraction to get wrong.
    """
    entity_col, key_col = links.columns[:2]
    link = links[[entity_col, key_col]].copy()
    link.columns = ["entity", "key"]
    if link_times is not None:
        link["link_time"] = np.asarray(link_times)
        # A membership whose formation time is unknown is dropped, not treated as having
        # always existed: `merge_asof` rejects nulls in its key, and the permissive reading
        # is exactly the future-visible behaviour this argument exists to remove.
        link = link.dropna(subset=["entity", "key", "link_time"])
        # Earliest formation wins: a membership exists from the first time it is recorded.
        link = link.sort_values("link_time", kind="stable").drop_duplicates(
            subset=["entity", "key"], keep="first")
    else:
        link = link.dropna().drop_duplicates()

    ready = pd.Series(np.asarray(label_times))
    if label_horizon is not None:
        ready = ready + label_horizon
    events = pd.DataFrame({"entity": np.asarray(label_entities),
                           "ready": ready.to_numpy(),
                           "y": np.asarray(label_values, dtype=np.float64)})

    columns = [f"{prefix}n_prior", f"{prefix}n_positive", f"{prefix}positive_rate",
               f"{prefix}n_linked"]
    out = pd.DataFrame({columns[0]: np.zeros(len(query_entities), dtype=np.int64),
                        columns[1]: np.full(len(query_entities), np.nan),
                        columns[2]: np.full(len(query_entities), np.nan),
                        columns[3]: np.zeros(len(query_entities), dtype=np.int64)})

    per_key = link.merge(events, on="entity", how="inner")
    if not len(per_key):
        return out
    if link_times is not None:
        # A neighbour's outcome is usable only once *both* facts are true: the outcome has
        # resolved, and the membership that connects it to this query already existed.
        # Taking the later of the two makes one as-of scan enforce both.
        per_key["ready"] = np.maximum(per_key["ready"].to_numpy(),
                                      per_key["link_time"].to_numpy())
    per_key = per_key.sort_values("ready", kind="stable")
    per_key["cum_y"] = per_key.groupby("key")["y"].cumsum()
    per_key["cum_n"] = per_key.groupby("key").cumcount() + 1

    queries = pd.DataFrame({"row": np.arange(len(query_entities)),
                            "entity": np.asarray(query_entities),
                            "cutoff": np.asarray(query_times)})
    expanded = queries.merge(link, on="entity", how="inner")
    if not len(expanded):
        return out

    # Pure structural degree: how many other entities share a key, with labels playing no
    # part whatsoever. This exists to be *compared against* ``n_prior``. If the two score
    # alike, the count feature is carrying structure rather than label timing, and no
    # leakage question arises for it -- a distinction that decided whether a rel-event
    # result was reportable.
    #
    # **It carries its own caveat, and it is not a label one.** The link table is used
    # whole, with no time filter, because most link tables carry no timestamp. Where a
    # membership *was* formed after the query's cutoff, this counts it -- structure from
    # the future. The temporal control does not detect this: that control moves label
    # cutoffs, and this quantity does not consult labels. On a task whose links are
    # timestamped, filter them before calling; on one whose links are not, any lift from
    # this column is an upper bound. rel-event is the latter.
    if link_times is None:
        sizes = link.groupby("key")["entity"].size()
        linked = (expanded.assign(n=expanded["key"].map(sizes).fillna(1) - 1)
                  .groupby("row")["n"].sum())
    else:
        # As-of degree: how many memberships of this key had formed by the cutoff. Without
        # this the column counts memberships created *after* the prediction time, which on
        # rel-event is worth several points of entirely spurious lift.
        ordered = link.sort_values("link_time", kind="stable")
        ordered["cum"] = ordered.groupby("key").cumcount() + 1
        asof = pd.merge_asof(
            expanded.sort_values("cutoff", kind="stable"),
            ordered[["key", "link_time", "cum"]].sort_values("link_time", kind="stable"),
            left_on="cutoff", right_on="link_time", by="key",
            direction="backward", allow_exact_matches=True,
        )
        # The query's own membership is inside that count whenever it had formed, and a
        # row must not count itself as one of its own neighbours.
        own = asof["link_time_x"].notna() & (asof["link_time_x"] <= asof["cutoff"])
        asof["n"] = (asof["cum"].fillna(0) - own.astype(float)).clip(lower=0)
        linked = asof.groupby("row")["n"].sum()
    out.loc[linked.index.to_numpy(), columns[3]] = linked.to_numpy().astype(np.int64)
    matched = pd.merge_asof(
        expanded.sort_values("cutoff", kind="stable"),
        per_key[["key", "ready", "cum_y", "cum_n"]].sort_values("ready", kind="stable"),
        left_on="cutoff", right_on="ready", by="key",
        direction="backward", allow_exact_matches=True,
    ).dropna(subset=["cum_n"])
    if not len(matched):
        return out

    # Subtract this row's own outcome wherever it falls inside its own window.
    own = events.groupby("entity", as_index=False).agg(own_ready=("ready", "min"),
                                                       own_y=("y", "first"))
    matched = matched.merge(own, on="entity", how="left")
    inside = matched["own_ready"].notna() & (matched["own_ready"] <= matched["cutoff"])
    matched["cum_n"] = matched["cum_n"] - inside.astype(float)
    matched["cum_y"] = matched["cum_y"] - np.where(inside, matched["own_y"].fillna(0.0), 0.0)

    agg = matched.groupby("row")[["cum_y", "cum_n"]].sum()
    rows = agg.index.to_numpy()
    n_prior = agg["cum_n"].to_numpy()
    n_pos = agg["cum_y"].to_numpy()
    out.loc[rows, columns[0]] = n_prior
    out.loc[rows, columns[1]] = np.where(n_prior > 0, n_pos, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out.loc[rows, columns[2]] = np.where(n_prior > 0, n_pos / n_prior, np.nan)
    out[columns[0]] = out[columns[0]].astype(np.int64)
    return out


def _one_hop_pairs(
    edges: np.ndarray, query_nodes: np.ndarray, label_nodes: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Unique ``(query_node, labelled_neighbour)`` pairs, self-loops removed.

    Filtering to query rows and labelled endpoints *before* materialising anything is what
    keeps this affordable: rel-event's friendship list is 30.4M edges, but only the ones
    touching a query at one end and a labelled user at the other can contribute.
    """
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be (n_edges, 2)")

    is_query = np.zeros(n_nodes, dtype=bool)
    is_query[query_nodes] = True
    is_labelled = np.zeros(n_nodes, dtype=bool)
    is_labelled[label_nodes] = True

    both = np.concatenate([edges, edges[:, ::-1]])
    both = both[both[:, 0] != both[:, 1]]                 # a node is never its own neighbour
    keep = is_query[both[:, 0]] & is_labelled[both[:, 1]]
    pairs = both[keep]
    if not len(pairs):
        return pairs.reshape(0, 2)
    return np.unique(pairs, axis=0)


def neighbour_label_features(
    edges: np.ndarray,
    label_nodes: np.ndarray,
    label_values: np.ndarray,
    query_nodes: np.ndarray,
    label_times: Optional[np.ndarray] = None,
    query_times: Optional[np.ndarray] = None,
    label_horizon: Optional[pd.Timedelta] = None,
    n_nodes: Optional[int] = None,
    prefix: str = "nbr__",
) -> pd.DataFrame:
    """One-hop neighbour-label aggregates, one row per query.

    Parameters
    ----------
    edges : np.ndarray
        ``(n_edges, 2)`` over a shared node id space. Treated as undirected.

    label_nodes, label_values : np.ndarray
        The **training** label events: which node, and its outcome. One node may appear
        several times, once per prediction time. Never pass validation or test labels --
        that is not a leak this module can detect, because such labels are indistinguishable
        from training ones once they are here.

    query_nodes : np.ndarray
        Node id per query row. May include nodes that also appear in ``label_nodes``; a row
        still never sees its own label, because a node is not its own neighbour.

    label_times, query_times : np.ndarray, optional
        Timestamps for the label events and the queries. Supply both to restrict each query
        to labels already resolved at its cutoff. Omitting them uses every training label
        regardless of time, which is faster, and wrong wherever the split is temporal.

    label_horizon : pd.Timedelta, optional
        How long after its own prediction time a label takes to resolve -- RelBench's
        ``task.timedelta``. A neighbour's outcome becomes usable at ``label_time +
        label_horizon``, not at ``label_time``. Without this the features read outcomes that
        were still being decided at the query's cutoff.

    n_nodes : int, optional
        Size of the node id space. Inferred from the inputs when omitted.

    prefix : str, default="nbr__"
        Column name prefix.

    Returns
    -------
    pd.DataFrame
        One row per query, in input order, with columns

        ``{prefix}labelled_degree``
            Neighbours holding a usable label. 0 is meaningful: no labelled neighbours.
        ``{prefix}positive_count``, ``{prefix}positive_rate``
            Positives among them, and their fraction. **NaN when the degree is 0**, so an
            isolated row is not scored as a confident negative.

    Notes
    -----
    Each neighbour contributes once, via its most recent usable label -- not once per label
    event. Otherwise a neighbour appearing under many prediction times would count many
    times, and the feature would measure activity rather than label.
    """
    query_nodes = np.asarray(query_nodes)
    label_nodes = np.asarray(label_nodes)
    label_values = np.asarray(label_values, dtype=np.float64)
    if len(label_nodes) != len(label_values):
        raise ValueError("label_nodes and label_values must be the same length")
    if (label_times is None) != (query_times is None):
        raise ValueError("supply both label_times and query_times, or neither")
    if label_horizon is not None and label_times is None:
        raise ValueError("label_horizon is meaningless without timestamps")

    edges = np.asarray(edges)
    if n_nodes is None:
        n_nodes = int(max(edges.max(initial=0), query_nodes.max(initial=0),
                          label_nodes.max(initial=0))) + 1

    columns = [f"{prefix}labelled_degree", f"{prefix}positive_count", f"{prefix}positive_rate"]
    empty = pd.DataFrame(
        {columns[0]: np.zeros(len(query_nodes), dtype=np.int64),
         columns[1]: np.full(len(query_nodes), np.nan),
         columns[2]: np.full(len(query_nodes), np.nan)}
    )

    pairs = _one_hop_pairs(edges, query_nodes, label_nodes, n_nodes)
    if not len(pairs):
        return empty

    # Expand query rows against their neighbours. Keyed by node, so every row belonging to
    # a node picks up that node's neighbours -- with its own cutoff applied below.
    queries = pd.DataFrame({"row": np.arange(len(query_nodes)), "node": query_nodes})
    if query_times is not None:
        queries["cutoff"] = np.asarray(query_times)
    joined = queries.merge(pd.DataFrame(pairs, columns=["node", "nbr"]), on="node", how="inner")
    if not len(joined):
        return empty

    events = pd.DataFrame({"nbr": label_nodes, "y": label_values})
    if label_times is None:
        # One value per node, the mean over its label events, then a plain join.
        per_node = events.groupby("nbr", as_index=False)["y"].mean()
        matched = joined.merge(per_node, on="nbr", how="inner")
    else:
        events["ready"] = np.asarray(label_times)
        if label_horizon is not None:
            events["ready"] = events["ready"] + label_horizon
        # merge_asof takes the latest label per neighbour that is already resolved at the
        # query's cutoff. Exact matches count: a window closing exactly at the cutoff is
        # known. Both sides must be sorted on the key merge_asof reads.
        left = joined.dropna(subset=["cutoff"]).sort_values("cutoff", kind="stable")
        right = events.sort_values("ready", kind="stable")
        matched = pd.merge_asof(
            left, right, left_on="cutoff", right_on="ready", by="nbr",
            direction="backward", allow_exact_matches=True,
        )
        matched = matched.dropna(subset=["y"])

    if not len(matched):
        return empty

    agg = matched.groupby("row")["y"].agg(["count", "sum"])
    out = empty.copy()
    rows = agg.index.to_numpy()
    degree = agg["count"].to_numpy()
    positives = agg["sum"].to_numpy()
    out.loc[rows, columns[0]] = degree
    out.loc[rows, columns[1]] = positives
    with np.errstate(invalid="ignore", divide="ignore"):
        out.loc[rows, columns[2]] = np.where(degree > 0, positives / degree, np.nan)
    out[columns[0]] = out[columns[0]].astype(np.int64)
    return out
