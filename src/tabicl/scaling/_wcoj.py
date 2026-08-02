"""Worst-case optimal joins, for relational features that live on cycles.

``_relational.py`` aggregates over a join *tree*: every child has one parent, so
statistics roll up by addition and nothing is counted twice. Cyclic patterns break
that. A triangle "users who share two mutual counterparties" has no root to roll up
from, and computing it with binary joins materialises an intermediate that can be
quadratically larger than the answer.

Leapfrog triejoin avoids the blow-up. It runs all relations forward in lockstep over
one variable at a time, so its cost is bounded by the AGM bound of the query -- the
size the output could have -- rather than by the size of any intermediate.

References
----------
.. [1] Veldhuizen. "Triejoin: A Simple, Worst-Case Optimal Join Algorithm." ICDT 2014.
.. [2] Ngo, Porat, Re, Rudra. "Worst-case Optimal Join Algorithms." PODS 2012.
.. [3] Abo Khamis, Ngo, Rudra. "FAQ: Questions Asked Frequently." PODS 2016.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # optional compiled accelerator; see build_native.py
    from . import _wcoj_native  # type: ignore
except ImportError:  # pragma: no cover - depends on whether it was built
    _wcoj_native = None

__all__ = [
    "Atom",
    "wcoj_join",
    "wcoj_count",
    "triangle_counts",
    "motif_features",
    "native_available",
]


def native_available() -> bool:
    """Whether the compiled hash-join backend is importable."""
    return _wcoj_native is not None


@dataclass
class Atom:
    """One relation in a conjunctive query.

    Parameters
    ----------
    name : str
        Label, used only in error messages.

    variables : Sequence[str]
        Variable bound to each column, e.g. ``("a", "b")`` for an edge table.

    data : np.ndarray
        Integer array of shape ``(n_rows, len(variables))``. Values are treated as
        opaque identifiers; use :func:`~pandas.factorize` for non-integer keys.
    """

    name: str
    variables: Sequence[str]
    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data)
        if self.data.ndim != 2 or self.data.shape[1] != len(self.variables):
            raise ValueError(
                f"atom {self.name!r}: data has shape {self.data.shape}, "
                f"expected (n, {len(self.variables)})"
            )


def _sorted_for(atom: Atom, order: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """Reorder an atom's columns to follow ``order``, then sort lexicographically.

    Leapfrog triejoin requires every relation to be a trie over the *global* variable
    ordering restricted to its own variables -- otherwise the coordinated seek at
    depth d is not comparing the same variable across relations.
    """
    local = [v for v in order if v in atom.variables]
    cols = [list(atom.variables).index(v) for v in local]
    data = atom.data[:, cols]
    if data.shape[0] > 1:
        # lexsort takes keys last-significant-first
        data = data[np.lexsort(tuple(data[:, i] for i in range(data.shape[1] - 1, -1, -1)))]
    return data, local


def _leapfrog(cursors: List[int], ends: List[int], cols: List[np.ndarray]) -> Tuple[int, bool]:
    """Advance cursors until every relation sits on the same value.

    This is the heart of the algorithm: repeatedly take the largest current key and
    seek every other relation up to it. A relation that cannot reach it is exhausted,
    which ends the whole variable. Each seek is a binary search, so a relation is
    never scanned past values that cannot join.
    """
    k = len(cursors)
    for i in range(k):
        if cursors[i] >= ends[i]:
            return 0, False

    x = max(int(cols[i][cursors[i]]) for i in range(k))
    matched = 0
    i = 0
    while matched < k:
        cur = int(cols[i][cursors[i]])
        if cur == x:
            matched += 1
        else:
            # Seek relation i forward to the first value >= x.
            seg = cols[i][cursors[i] : ends[i]]
            cursors[i] += int(np.searchsorted(seg, x, side="left"))
            if cursors[i] >= ends[i]:
                return 0, False
            cur = int(cols[i][cursors[i]])
            if cur > x:
                x, matched = cur, 1
            else:
                matched += 1
        i = (i + 1) % k
    return x, True


def wcoj_join(
    atoms: Sequence[Atom],
    order: Sequence[str],
    max_results: int | None = None,
    less_than: Sequence[Tuple[str, str]] = (),
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    """Evaluate a conjunctive query with leapfrog triejoin.

    Parameters
    ----------
    atoms : Sequence[Atom]
        The relations. Repeating the same table under different variables is how
        cyclic patterns are expressed -- a triangle is three copies of ``edge``.

    order : Sequence[str]
        Global variable ordering. Cost depends on it; put high-selectivity variables
        first.

    max_results : int, optional
        Stop after this many tuples. Useful as a guard when a pattern is more
        frequent than expected.

    less_than : Sequence[tuple[str, str]], default=()
        Pairs ``(u, v)`` constraining ``u < v``. Enforced *inside* the join rather
        than by filtering afterwards, which is what makes symmetry breaking pay: an
        undirected triangle has 6 orientations, and requiring ``a < b < c`` stops the
        other 5 from ever being enumerated. ``u`` must precede ``v`` in ``order``.

    threads : int, default=0
        Worker threads for the compiled backend; 0 means one per core. Ignored by the
        Python backend and when ``max_results`` is set, since capping makes *which*
        tuples are returned order-dependent.

    backend : {"auto", "native", "python"}, default="auto"
        ``"auto"`` uses the compiled hash-join backend when it was built and falls
        back to the pure-Python leapfrog triejoin otherwise. The two return the same
        tuples; only the row order may differ, so compare as sets.

    Returns
    -------
    np.ndarray
        Shape ``(n_results, len(order))``, one column per variable in ``order``.

    Examples
    --------
    >>> import numpy as np
    >>> e = np.array([[1, 2], [2, 3], [1, 3]])
    >>> tri = wcoj_join(
    ...     [Atom("e", ("a", "b"), e), Atom("e", ("b", "c"), e), Atom("e", ("a", "c"), e)],
    ...     ["a", "b", "c"],
    ... )
    >>> tri.tolist()
    [[1, 2, 3]]
    """
    order = list(order)

    if backend not in ("auto", "native", "python"):
        raise ValueError(f"backend must be auto/native/python, got {backend!r}")
    if backend == "native" and _wcoj_native is None:
        raise RuntimeError("native backend requested but not built; see build_native.py")
    if backend != "python" and _wcoj_native is not None:
        return _native_join(atoms, order, max_results, less_than, threads)

    prepared = []
    for atom in atoms:
        data, local = _sorted_for(atom, order)
        missing = set(atom.variables) - set(order)
        if missing:
            raise ValueError(f"atom {atom.name!r} uses variables not in order: {sorted(missing)}")
        prepared.append((data, local))

    # Which atoms constrain each variable, and at which of their columns.
    by_var: Dict[str, List[Tuple[int, int]]] = {v: [] for v in order}
    for idx, (_data, local) in enumerate(prepared):
        for depth, var in enumerate(local):
            by_var[var].append((idx, depth))

    for var, participants in by_var.items():
        if not participants:
            raise ValueError(f"variable {var!r} appears in no atom")

    # For each level, which earlier levels it must strictly exceed.
    position = {v: i for i, v in enumerate(order)}
    lower_bounds: List[List[int] | None] = [None] * len(order)
    for lesser, greater in less_than:
        if lesser not in position or greater not in position:
            raise ValueError(f"less_than references unknown variable: {(lesser, greater)}")
        if position[lesser] >= position[greater]:
            raise ValueError(
                f"less_than {(lesser, greater)} requires {lesser!r} to precede "
                f"{greater!r} in order"
            )
        slot = position[greater]
        lower_bounds[slot] = (lower_bounds[slot] or []) + [position[lesser]]

    results: List[List[int]] = []
    binding = [0] * len(order)
    # Row range each atom is currently confined to, narrowed as variables are bound.
    ranges = [[(0, len(data))] for data, _ in prepared]

    def recurse(level: int) -> bool:
        """Bind ``order[level]``; return False to signal the result cap was hit."""
        if level == len(order):
            results.append(list(binding))
            return max_results is None or len(results) < max_results

        var = order[level]
        participants = by_var[var]
        cursors, ends, cols = [], [], []
        for atom_idx, depth in participants:
            lo, hi = ranges[atom_idx][-1]
            cursors.append(lo)
            ends.append(hi)
            cols.append(prepared[atom_idx][0][:, depth])

        # Symmetry breaking: skip straight past values that cannot satisfy an
        # ordering constraint, instead of generating them and discarding them later.
        floor = lower_bounds[level]
        if floor is not None:
            bound = max(binding[j] for j in floor) + 1
            for slot in range(len(cursors)):
                seg = cols[slot][cursors[slot] : ends[slot]]
                cursors[slot] += int(np.searchsorted(seg, bound, side="left"))

        while True:
            value, ok = _leapfrog(cursors, ends, cols)
            if not ok:
                return True

            # Narrow each participating atom to the rows carrying this value, so the
            # next variable only sees consistent extensions.
            for slot, (atom_idx, _depth) in enumerate(participants):
                col = cols[slot]
                start = cursors[slot]
                seg = col[start : ends[slot]]
                stop = start + int(np.searchsorted(seg, value, side="right"))
                ranges[atom_idx].append((start, stop))

            binding[level] = value
            keep_going = recurse(level + 1)

            for atom_idx, _depth in participants:
                ranges[atom_idx].pop()
            if not keep_going:
                return False

            # Step every participant past this value.
            for slot in range(len(cursors)):
                seg = cols[slot][cursors[slot] : ends[slot]]
                cursors[slot] += int(np.searchsorted(seg, value, side="right"))

    recurse(0)
    if not results:
        return np.empty((0, len(order)), dtype=np.int64)
    return np.asarray(results, dtype=np.int64)


def _native_join(
    atoms: Sequence[Atom],
    order: List[str],
    max_results: int | None,
    less_than: Sequence[Tuple[str, str]],
    threads: int = 0,
) -> np.ndarray:
    """Marshal to the compiled backend and enumerate."""
    relations, var_ids, pairs = _marshal(atoms, order, less_than)
    return _wcoj_native.wcoj_hash_join(
        relations, var_ids, len(order), pairs, 0 if max_results is None else int(max_results), int(threads)
    )


def _marshal(atoms: Sequence[Atom], order: List[str], less_than: Sequence[Tuple[str, str]]):
    """Permute columns into global variable order and translate names to indices."""
    position = {v: i for i, v in enumerate(order)}
    relations, var_ids = [], []
    for atom in atoms:
        missing = set(atom.variables) - set(order)
        if missing:
            raise ValueError(f"atom {atom.name!r} uses variables not in order: {sorted(missing)}")
        local = [v for v in order if v in atom.variables]
        cols = [list(atom.variables).index(v) for v in local]
        relations.append(np.ascontiguousarray(atom.data[:, cols], dtype=np.int64))
        var_ids.append([position[v] for v in local])

    pairs = []
    for lesser, greater in less_than:
        if lesser not in position or greater not in position:
            raise ValueError(f"less_than references unknown variable: {(lesser, greater)}")
        if position[lesser] >= position[greater]:
            raise ValueError(
                f"less_than {(lesser, greater)} requires {lesser!r} to precede {greater!r} in order"
            )
        pairs.append((position[lesser], position[greater]))
    return relations, var_ids, pairs


def wcoj_count(
    atoms: Sequence[Atom],
    order: Sequence[str],
    less_than: Sequence[Tuple[str, str]] = (),
    threads: int = 0,
) -> Tuple[int, np.ndarray]:
    """Count results of a conjunctive query without materialising any of them.

    This is the FAQ view of the query (Abo Khamis, Ngo, Rudra, PODS 2016): aggregation
    is variable elimination over a semiring, so the count is accumulated *during* the
    join rather than by enumerating tuples and counting afterwards. Once a prefix is
    fixed, the number of completions is the size of an intersection -- known without
    visiting its elements -- so every already-bound variable is credited in O(1).

    Requires the compiled backend; there is no Python fallback, because a Python
    implementation would be slower than enumerating.

    Parameters
    ----------
    atoms, order, less_than
        As for :func:`wcoj_join`.

    threads : int, default=0
        Worker threads; 0 means one per core.

    Returns
    -------
    total : int
        Number of result tuples.

    occurrences : np.ndarray
        ``occurrences[v]`` is how many result tuples contain ``v`` in any position,
        indexed by value. Length is one past the largest value in ``atoms``.
    """
    if _wcoj_native is None:
        raise RuntimeError("wcoj_count needs the compiled backend; see build_native.py")
    order = list(order)
    relations, var_ids, pairs = _marshal(atoms, order, less_than)
    total, counts = _wcoj_native.wcoj_count(relations, var_ids, len(order), pairs, int(threads))
    return int(total), counts


def _undirected_edges(edges: np.ndarray) -> np.ndarray:
    """Symmetrise and dedupe, so an undirected motif is counted once per orientation."""
    both = np.vstack([edges, edges[:, ::-1]])
    both = both[both[:, 0] != both[:, 1]]
    return np.unique(both, axis=0)


def triangle_counts(edges: np.ndarray, nodes: np.ndarray | None = None) -> np.ndarray:
    """Count triangles through each node.

    Prefers the compiled join when it has been built, and falls back to sparse
    ``diag(A^3)/2`` otherwise. Measured on this machine, the compiled join beats
    ``scipy`` by 2-12x on uniform random graphs and by ~27x on hub-skewed ones, where
    ``A^3`` densifies; the pure-Python join loses to everything and is never chosen
    here. See ``DESIGN.md`` for the table.

    Parameters
    ----------
    edges : np.ndarray
        Shape ``(n_edges, 2)`` integer array. Treated as undirected.

    nodes : np.ndarray, optional
        Nodes to report, in order. Defaults to every node appearing in ``edges``.

    Returns
    -------
    np.ndarray
        Triangle count per node, aligned with ``nodes``.
    """
    e = _undirected_edges(np.asarray(edges))
    if nodes is None:
        nodes = np.unique(e) if e.size else np.empty(0, dtype=np.int64)
    nodes = np.asarray(nodes)

    if e.size == 0 or len(nodes) == 0:
        return np.zeros(len(nodes), dtype=np.int64)

    if _wcoj_native is not None:
        # a < b < c is pushed into the join, so each triangle is counted once; the
        # count is accumulated during elimination, so no triangle is ever built.
        _total, occurrences = wcoj_count(
            [Atom("e", ("a", "b"), e), Atom("e", ("b", "c"), e), Atom("e", ("a", "c"), e)],
            ["a", "b", "c"],
            less_than=[("a", "b"), ("b", "c")],
        )
        in_range = nodes < len(occurrences)
        counts = np.zeros(len(nodes), dtype=np.int64)
        counts[in_range] = occurrences[nodes[in_range]]
        return counts

    import scipy.sparse as sp

    size = int(max(e.max(), nodes.max())) + 1
    adj = sp.csr_matrix(
        (np.ones(len(e), dtype=np.int64), (e[:, 0], e[:, 1])), shape=(size, size)
    )
    # diag(A^3)[i] counts closed walks of length 3 through i, i.e. each of its
    # triangles once per direction -- hence the halving.
    return ((adj @ adj @ adj).diagonal() // 2)[nodes].astype(np.int64)


def motif_features(edges: np.ndarray, nodes: np.ndarray | None = None):
    """Per-node graph features for a cyclic relationship, as a DataFrame.

    Produces degree, triangle count, and clustering coefficient -- the last being the
    fraction of a node's neighbour pairs that are themselves connected, which is
    exactly a triangle-versus-wedge ratio and therefore needs the cyclic count.

    Parameters
    ----------
    edges : np.ndarray
        Shape ``(n_edges, 2)``. Treated as undirected.

    nodes : np.ndarray, optional
        Nodes to report, in order.

    Returns
    -------
    pd.DataFrame
        Indexed by node, with ``degree``, ``triangles`` and ``clustering``.
    """
    import pandas as pd

    e = _undirected_edges(np.asarray(edges))
    if nodes is None:
        nodes = np.unique(e) if e.size else np.empty(0, dtype=np.int64)
    nodes = np.asarray(nodes)

    degree = np.zeros(len(nodes), dtype=np.int64)
    if e.size:
        members, freq = np.unique(e[:, 0], return_counts=True)
        index = {int(n): i for i, n in enumerate(nodes)}
        for node, count in zip(members, freq):
            slot = index.get(int(node))
            if slot is not None:
                degree[slot] = count

    triangles = triangle_counts(e, nodes)
    wedges = degree * (degree - 1) / 2
    clustering = np.divide(triangles, wedges, out=np.zeros(len(nodes)), where=wedges > 0)

    return pd.DataFrame(
        {"degree": degree, "triangles": triangles, "clustering": clustering},
        index=pd.Index(nodes, name="node"),
    )
