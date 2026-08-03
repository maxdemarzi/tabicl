"""Choosing *which* rows go in context, rather than how to fit all of them.

Every memory technique here so far -- chunking, offloading, a smaller KV cache --
accepts that the whole training set is the context and tries to survive it. Retrieval
questions the premise: TabDPT-style models put only the nearest neighbours of the
query in context, so cost scales with the neighbourhood rather than the dataset.

TabICL already faces this question, because it subsamples above its context limit.
It currently answers it at random. This module supplies the alternative, so the two
can be compared at *equal context size* -- which is the only comparison that isolates
selection quality from context length.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["select_context"]


def select_context(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_context: int,
    method: str = "knn",
    random_state: Optional[int] = 0,
    per_query: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick ``n_context`` training rows to serve as in-context examples.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        The candidate pool.

    X_test : np.ndarray
        Rows to be predicted. Used only to locate the relevant region; labels are
        never involved, so this leaks nothing.

    n_context : int
        Size of the returned context. Both methods return exactly this many rows
        (or the whole pool if it is smaller), so a comparison between them varies
        selection and holds context length fixed.

    method : {"knn", "random"}, default="knn"
        ``"knn"`` takes the union of each test row's nearest neighbours, which
        concentrates the context where the queries actually are. ``"random"`` is the
        uniform subsample TabICL does today, kept here as the control.

    random_state : int, optional
        Seed for the random method and for tie-breaking top-ups.

    per_query : int, default=8
        Neighbours drawn per test row before deduplication. Larger values cover the
        query distribution more evenly but saturate ``n_context`` sooner, so the
        union is truncated by frequency: rows that are a neighbour of many queries
        are kept first.

    Returns
    -------
    X_context, y_context : np.ndarray
        The selected rows and their labels.
    """
    n_train = len(X_train)
    if n_context >= n_train:
        return X_train, y_train
    rng = np.random.default_rng(random_state)

    if method == "random":
        idx = rng.choice(n_train, size=n_context, replace=False)
        return X_train[idx], y_train[idx]

    if method != "knn":
        raise ValueError(f"method must be 'knn' or 'random', got {method!r}")

    from sklearn.neighbors import NearestNeighbors

    # Standardise before measuring distance: raw scales would let one wide column
    # dictate the neighbourhood.
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    tr = (X_train - mu) / sigma
    te = (X_test - mu) / sigma

    k = min(per_query, n_train)
    nn = NearestNeighbors(n_neighbors=k).fit(tr)
    neighbours = nn.kneighbors(te, return_distance=False)

    # A row that is a neighbour of many queries is more useful than one that serves a
    # single query, so rank the union by how often it was retrieved.
    counts = np.bincount(neighbours.ravel(), minlength=n_train)
    order = np.argsort(-counts, kind="stable")
    chosen = order[:n_context]

    # If fewer distinct rows were retrieved than requested, top up at random rather
    # than returning a shorter context, so the comparison stays like-for-like.
    n_retrieved = int((counts > 0).sum())
    if n_retrieved < n_context:
        rest = order[n_retrieved:]
        extra = rng.permutation(rest)[: n_context - n_retrieved]
        chosen = np.concatenate([order[:n_retrieved], extra])

    return X_train[chosen], y_train[chosen]
