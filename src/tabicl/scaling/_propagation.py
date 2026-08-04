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

__all__ = ["neighbour_label_features"]


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
