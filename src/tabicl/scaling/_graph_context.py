"""Graph neighbours as in-context examples: message passing without touching weights.

`RESEARCH.md` item 8. TabICL attends from each test row to labelled context rows. If the
context holds that row's **graph neighbours with their labels**, then attention is one
round of learned message passing — structurally what a GNN layer does, obtained with no
weight change and no retraining.

This is the only idea in that file that attacks the i.i.d. limitation directly. Flattening
can aggregate a neighbour's *features* but never its *label*, because at fit time using
those labels is leakage or transduction and at predict time they are unknown. Putting
neighbours in the **context** sidesteps that: context rows are labelled training data by
construction, which is what an in-context learner is for.

It also reframes retrieval usefully. Retrieval by *feature* similarity disappointed both
here and in an independent four-dataset evaluation. Graph proximity is a different
hypothesis, and unlike feature similarity it targets the mechanism that GNNs win on.

Budget comes free: rel-avito matches full-context quality on 8.6% of its rows, so there is
room to choose *which* rows without paying more.

**Check `label_homophily` before building anything.** If a node's label is uncorrelated
with its neighbours', no context construction can help, and it is one cheap number.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["label_homophily", "select_graph_context"]


def label_homophily(edges: np.ndarray, labels: np.ndarray) -> dict:
    """Do connected nodes share labels more than chance?

    The gate for everything else in this module. Returns edge homophily -- the fraction
    of edges joining same-label nodes -- alongside the fraction expected if labels were
    assigned at random, so the two are comparable directly.

    Parameters
    ----------
    edges : np.ndarray
        ``(n_edges, 2)`` integer array indexing into ``labels``. Treated as undirected.

    labels : np.ndarray
        One label per node. Nodes whose label is unknown must be masked out by the caller
        before this is called, or they will be counted as a class of their own.

    Returns
    -------
    dict
        ``observed`` (fraction of same-label edges), ``expected`` (the same under random
        assignment, ``sum p_c^2``), ``lift`` (observed − expected), and ``n_edges``.

    Notes
    -----
    ``lift`` near zero is a stop signal: it means neighbours carry no information about a
    node's label, so neither k-hop counts nor neighbour-context can help. A negative lift
    is *heterophily*, which is real signal but needs the opposite treatment.
    """
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be (n_edges, 2)")
    if not len(edges):
        return {"observed": float("nan"), "expected": float("nan"),
                "lift": float("nan"), "n_edges": 0}

    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    observed = float(same.mean())

    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    expected = float((proportions ** 2).sum())

    return {"observed": observed, "expected": expected,
            "lift": observed - expected, "n_edges": int(len(edges))}


def select_graph_context(
    edges: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    n_context: int,
    hops: int = 1,
    random_state: Optional[int] = 0,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Choose context rows by graph proximity to the queries.

    Parameters
    ----------
    edges : np.ndarray
        ``(n_edges, 2)`` over a shared node id space. Undirected.

    train_idx : np.ndarray
        Node ids eligible for the context, i.e. rows whose labels may be used. Query rows
        must not appear here -- a query in its own context is the leak this whole family
        is prone to, so it is filtered rather than trusted.

    query_idx : np.ndarray
        Node ids to be predicted.

    n_context : int
        Context size. Neighbours are taken first, ranked by how many queries they are a
        neighbour of; any shortfall is filled at random from the remaining pool so the
        returned context is always exactly this size. That matters because the comparison
        against random selection is only meaningful at equal context length.

    hops : int, default=1
        Neighbourhood radius. Beyond 2 the reachable set saturates toward the whole
        component on a small-world graph, at which point every query has the same
        neighbourhood and the selection stops being a selection.

    random_state : int, optional
        Seed for the top-up.

    labels : np.ndarray, optional
        Label per node, used only to **stratify** the selection. Strongly recommended.

        Ranking by reach is a popularity ranking, and on a social graph popularity is
        correlated with the label: measured on rel-event, an unstratified selection
        returned a context with a positive rate of 0.019--0.046 against a base rate of
        0.163, i.e. it silently built a near-single-class context. Passing ``labels``
        keeps each class's share of the eligible pool intact and applies the proximity
        ranking *within* each class, so the method selects on graph structure rather than
        on the label through it.

    Returns
    -------
    np.ndarray
        Selected node ids from ``train_idx``, length ``min(n_context, len(train_idx))``.
    """
    if hops < 1:
        raise ValueError("hops must be at least 1")
    rng = np.random.default_rng(random_state)
    train_idx = np.asarray(train_idx)
    eligible = np.setdiff1d(train_idx, np.asarray(query_idx))   # never a query
    if n_context >= len(eligible):
        return eligible

    n_nodes = int(max(edges.max(initial=0), train_idx.max(initial=0),
                      np.asarray(query_idx).max(initial=0))) + 1
    # Symmetric adjacency as a sorted CSR-ish structure, built once.
    both = np.concatenate([edges, edges[:, ::-1]])
    order = np.argsort(both[:, 0], kind="stable")
    src, dst = both[order, 0], both[order, 1]
    starts = np.searchsorted(src, np.arange(n_nodes + 1))

    # Expand the frontier `hops` times, counting how many queries reach each node. A node
    # that many queries reach is more useful than one serving a single query, which is
    # what breaks ties when the neighbourhood overflows the budget.
    reached = np.zeros(n_nodes, dtype=np.int64)
    frontier = np.unique(np.asarray(query_idx))
    for _ in range(hops):
        neighbours = np.concatenate(
            [dst[starts[node]:starts[node + 1]] for node in frontier]
        ) if len(frontier) else np.empty(0, dtype=np.int64)
        if not len(neighbours):
            break
        np.add.at(reached, neighbours, 1)
        frontier = np.unique(neighbours)

    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) < n_nodes:
            raise ValueError(f"labels must cover {n_nodes} nodes, got {len(labels)}")
        eligible_labels = labels[eligible]
        classes, counts = np.unique(eligible_labels, return_counts=True)
        # Quota per class = that class's share of the eligible pool, so the context's
        # balance matches what random selection would have given.
        exact = counts / counts.sum() * n_context
        quota = np.floor(exact).astype(np.int64)
        while quota.sum() < n_context:                    # hand out the remainder
            quota[np.argmax(exact - quota)] += 1
        picked = []
        for cls, want in zip(classes, quota):
            pool = eligible[eligible_labels == cls]
            # Rank within the class by reach; ties broken at random rather than by node
            # id, which would otherwise bias every seed toward the same low-id nodes.
            order = np.lexsort((rng.permutation(len(pool)), -reached[pool]))
            picked.append(pool[order[:want]])
        return np.concatenate(picked).astype(np.int64)

    eligible_mask = np.zeros(n_nodes, dtype=bool)
    eligible_mask[eligible] = True
    scores = np.where(eligible_mask, reached, -1)

    ranked = np.argsort(-scores, kind="stable")
    chosen = ranked[:n_context]

    # Fill from the eligible pool if fewer nodes were reached than the budget, so the
    # context is always the requested size.
    n_reached = int((scores > 0).sum())
    if n_reached < n_context:
        rest = np.setdiff1d(eligible, ranked[:n_reached], assume_unique=False)
        extra = rng.permutation(rest)[: n_context - n_reached]
        chosen = np.concatenate([ranked[:n_reached], extra])

    return chosen.astype(np.int64)
