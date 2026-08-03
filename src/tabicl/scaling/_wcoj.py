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

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ._semiring import MAX_PLUS, MIN_PLUS, SUM_PRODUCT, Semiring

try:  # optional compiled accelerator; see build_native.py
    from . import _wcoj_native  # type: ignore
except ImportError:  # pragma: no cover - depends on whether it was built
    _wcoj_native = None

__all__ = [
    "Atom",
    "wcoj_join",
    "wcoj_count",
    "wcoj_aggregate",
    "triangle_counts",
    "motif_features",
    "typed_triangle_counts",
    "typed_motif_features",
    "temporal_motif_features",
    "typed_temporal_motif_features",
    "native_available",
]

# k edge types need k**3 joins for the full typed triangle census. The guard is about
# the cube, not the joins: 6 types is 216, 8 would be 512.
MAX_EDGE_TYPES = 6


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


def _require_non_negative(relations: Sequence[np.ndarray]) -> None:
    """Reject negative values before they reach an array index.

    The counting and aggregation kernels accumulate into arrays indexed *by value*,
    so a negative value is an out-of-bounds write rather than a wrong answer. This is
    not hypothetical: ``pd.factorize`` emits -1 for missing keys, so any real table
    with a null foreign key produces them.
    """
    for data in relations:
        if data.size and int(data.min()) < 0:
            raise ValueError(
                "values must be non-negative (they index the result arrays); "
                "pd.factorize emits -1 for missing keys, so drop or remap nulls first"
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
        data = atom.data
        if cols != list(range(len(cols))) or data.dtype != np.int64:
            # Fancy indexing copies, so only pay for it when the columns actually move.
            # Otherwise the three atoms of a triangle each get a private copy of the
            # same edge table, and the compiled backend cannot tell they are the same
            # buffer -- which is exactly what lets it build one trie instead of three.
            data = np.ascontiguousarray(data[:, cols], dtype=np.int64)
        elif not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        relations.append(data)
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
    _require_non_negative(relations)
    total, counts = _wcoj_native.wcoj_count(relations, var_ids, len(order), pairs, int(threads))
    return int(total), counts


_RING_CODES = {SUM_PRODUCT.name: 0, MIN_PLUS.name: 1, MAX_PLUS.name: 2}


def wcoj_aggregate(
    atoms: Sequence[Atom],
    order: Sequence[str],
    weights: np.ndarray,
    semiring: Semiring = SUM_PRODUCT,
    less_than: Sequence[Tuple[str, str]] = (),
    threads: int = 0,
) -> Tuple[int, float, np.ndarray]:
    """Aggregate a payload over every witness of a pattern, without enumerating them.

    Full FAQ: ``semiring.mul`` accumulates along a witness, ``semiring.add`` combines
    alternative witnesses, and the fold happens during variable elimination. Counting
    is the special case where every payload is 1.

    This is what tree aggregation cannot express -- a cyclic pattern has no root to
    roll up from, so "the cheapest triangle each node participates in" has no
    formulation as a parent-child roll-up at all.

    Parameters
    ----------
    atoms, order, less_than
        As for :func:`wcoj_join`.

    weights : np.ndarray
        Payload per *value*, indexed by value. Length must cover the largest value
        appearing in ``atoms``.

    semiring : Semiring, default=SUM_PRODUCT
        One of ``SUM_PRODUCT`` (sum over witnesses), ``MIN_PLUS`` (cheapest witness),
        ``MAX_PLUS`` (strongest witness). Custom semirings are not supported by the
        compiled kernel.

    threads : int, default=0
        Worker threads; 0 means one per core.

    Returns
    -------
    total : int
        Number of witnesses.

    overall : float
        The payload aggregated over all witnesses.

    per_value : np.ndarray
        ``per_value[v]`` aggregates over witnesses containing ``v``. Values in no
        witness hold the semiring's ``zero``.
    """
    if _wcoj_native is None:
        raise RuntimeError("wcoj_aggregate needs the compiled backend; see build_native.py")
    code = _RING_CODES.get(semiring.name)
    if code is None:
        raise ValueError(
            f"semiring {semiring.name!r} is not supported by the compiled kernel; "
            f"expected one of {sorted(_RING_CODES)}"
        )

    order = list(order)
    relations, var_ids, pairs = _marshal(atoms, order, less_than)
    _require_non_negative(relations)
    largest = max((int(r.max()) for r in relations if r.size), default=-1)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if len(weights) <= largest:
        raise ValueError(f"weights has length {len(weights)} but values reach {largest}")

    total, overall, per_value = _wcoj_native.wcoj_aggregate(
        relations, var_ids, len(order), pairs, code, weights.tolist(), int(threads)
    )
    return int(total), float(overall), per_value


def _undirected_edges(edges: np.ndarray) -> np.ndarray:
    """Symmetrise and dedupe, so an undirected motif is counted once per orientation.

    Dedup is on a packed ``(u << 32) | v`` key rather than ``np.unique(..., axis=0)``.
    The 2-D form sorts a void view of each row, which is several times slower than
    sorting int64 and is called once per (cutoff x window) on the temporal path. Falls
    back to the general form when ids do not fit in 32 bits.
    """
    both = np.vstack([edges, edges[:, ::-1]])
    both = both[both[:, 0] != both[:, 1]]
    if both.size == 0:
        return both
    if both.dtype != np.int64 or int(both.max()) >= (1 << 31) or int(both.min()) < 0:
        return np.unique(both, axis=0)
    packed = (both[:, 0] << np.int64(32)) | both[:, 1]
    keep = np.unique(packed)
    return np.column_stack([keep >> np.int64(32), keep & np.int64(0xFFFFFFFF)])


def _gather(by_value: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Look up a value-indexed array at ``nodes``, scoring out-of-range nodes zero."""
    out = np.zeros(len(nodes), dtype=np.int64)
    if len(by_value) == 0 or len(nodes) == 0:
        return out
    in_range = (nodes >= 0) & (nodes < len(by_value))
    out[in_range] = by_value[nodes[in_range]]
    return out


def _degrees(e: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Degree of each node in a symmetrised edge list.

    Vectorised rather than a dict lookup per node: this runs once per (cutoff, window)
    in the temporal path, so a Python loop over the node set is not affordable there.
    """
    degree = np.zeros(len(nodes), dtype=np.int64)
    if e.size == 0 or len(nodes) == 0:
        return degree
    members, freq = np.unique(e[:, 0], return_counts=True)
    slot = np.searchsorted(members, nodes)
    hit = (slot < len(members)) & (members[np.minimum(slot, len(members) - 1)] == nodes)
    degree[hit] = freq[slot[hit]]
    return degree


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
        return _gather(occurrences, nodes)

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

    degree = _degrees(e, nodes)
    triangles = triangle_counts(e, nodes)
    wedges = degree * (degree - 1) / 2
    clustering = np.divide(triangles, wedges, out=np.zeros(len(nodes)), where=wedges > 0)

    return pd.DataFrame(
        {"degree": degree, "triangles": triangles, "clustering": clustering},
        index=pd.Index(nodes, name="node"),
    )


# ---------------------------------------------------------------------------
# Typed motifs
# ---------------------------------------------------------------------------
#
# The ablation in DESIGN.md found degree carrying ~85% of the lift on rel-event,
# i.e. most of what an untyped triangle count contributes is degree in disguise.
# That is the expected result rather than a surprise: motif counts are partly
# *determined* by the degree sequence (Ginoza & Mugler, "Network motifs come in
# sets", 2010; Bhat et al., "Motif conservation laws for the configuration model",
# 2014), and a k-star count is a deterministic function of degree outright.
#
# Splitting counts by edge type is the standard way out, because a typed count is
# not recoverable from the untyped degree: Lichtenwalter & Chawla's vertex
# collocation profiles (WWW 2012; SpringerPlus 2014) make exactly this argument,
# that the discriminative power comes from distinguishing isomorphism classes with
# direction and relation type rather than from going to larger undirected motifs.


def typed_triangle_counts(
    edges_by_type: Mapping[str, np.ndarray],
    nodes: np.ndarray | None = None,
    threads: int = 0,
):
    """Count triangles per node, split by the *multiset* of edge types they use.

    A triangle on ``{a, b, c}`` occupies three slots -- ``(a,b)``, ``(b,c)``, ``(a,c)``
    under ``a < b < c`` -- and each slot's type is then determined. So running the join
    once per *ordered* type triple and bucketing the result by the sorted triple counts
    every triangle exactly once, with no double counting to correct for afterwards.
    That is ``k ** 3`` joins for ``k`` types, each over a subset of the edges.

    Parameters
    ----------
    edges_by_type : Mapping[str, np.ndarray]
        One ``(n_edges, 2)`` integer array per type, over a shared node id space.
        Each is treated as undirected. At most :data:`MAX_EDGE_TYPES` types.

    nodes : np.ndarray, optional
        Nodes to report, in order. Defaults to every node appearing in any type.

    threads : int, default=0
        Worker threads for the compiled backend; 0 means one per core.

    Returns
    -------
    pd.DataFrame
        Indexed by node, one ``tri__<t1>_<t2>_<t3>`` column per sorted type triple.
        Column order follows the order of ``edges_by_type``.

    Notes
    -----
    Types are assumed disjoint -- one type per edge. An edge listed under several
    types is counted under each combination it satisfies, which is well defined but
    means the columns no longer sum to the untyped triangle count.

    Examples
    --------
    >>> import numpy as np
    >>> friend = np.array([[0, 1], [1, 2]])
    >>> colleague = np.array([[0, 2]])
    >>> out = typed_triangle_counts({"f": friend, "c": colleague})
    >>> int(out.loc[0, "tri__f_f_c"])   # two friendships closed by a colleague edge
    1
    """
    import pandas as pd

    types = list(edges_by_type)
    if not types:
        raise ValueError("edges_by_type is empty; give at least one edge type")
    if len(types) > MAX_EDGE_TYPES:
        raise ValueError(
            f"{len(types)} edge types needs {len(types) ** 3} joins for the full typed "
            f"census; at most {MAX_EDGE_TYPES} are allowed. Merge rare types first."
        )

    prepared = [_undirected_edges(np.asarray(edges_by_type[t])) for t in types]
    if nodes is None:
        present = [e for e in prepared if e.size]
        nodes = np.unique(np.concatenate(present)) if present else np.empty(0, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)

    columns = _typed_census(prepared, types, nodes, threads)
    return pd.DataFrame(
        {"tri__" + "_".join(types[i] for i in combo): counts for combo, counts in columns.items()},
        index=pd.Index(nodes, name="node"),
    )


def _typed_census(
    prepared: Sequence[np.ndarray],
    types: Sequence[str],
    nodes: np.ndarray,
    threads: int,
    wanted: set | None = None,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """Run the ordered-triple loop over already-symmetrised edge arrays.

    Split out so the temporal path can ask for a *subset* of combinations. Combos whose
    three slots are all static do not change with the cutoff, so recomputing them once
    per prediction time is pure waste; ``wanted`` is how they get skipped.

    Each edge table is indexed once and the handle reused across all ``k ** 3`` queries.
    Without that the census rebuilds the same tries `k ** 2` times each -- and the trie
    is a full lexicographic sort, so on large relations that dominated the run.

    Returns a dict keyed by sorted index triple, so the caller owns column naming.
    """
    columns = {
        combo: np.zeros(len(nodes), dtype=np.int64)
        for combo in itertools.combinations_with_replacement(range(len(types)), 3)
    }

    # Once per census rather than once per query, but it must still happen: the counting
    # kernel accumulates into an array indexed BY VALUE, so a negative value is an
    # out-of-bounds write rather than a wrong answer.
    _require_non_negative(prepared)

    indexed = None
    if _wcoj_native is not None and len(nodes):
        # Every atom below is already in global variable order (a<b, b<c, a<c under
        # ["a","b","c"]), so no column permutation stands between the array and its
        # index -- which is what makes one handle per type reusable across the census.
        indexed = [
            _wcoj_native.Relation(np.ascontiguousarray(e, dtype=np.int64)) if e.size else None
            for e in prepared
        ]
    var_ids = [[0, 1], [1, 2], [0, 2]]
    pairs = [(0, 1), (1, 2)]

    for i, j, k in itertools.product(range(len(types)), repeat=3):
        key = tuple(sorted((i, j, k)))
        if wanted is not None and key not in wanted:
            continue
        ab, bc, ac = prepared[i], prepared[j], prepared[k]
        if ab.size == 0 or bc.size == 0 or ac.size == 0 or len(nodes) == 0:
            continue
        slot = columns[key]
        if indexed is not None:
            _total, occurrences = _wcoj_native.wcoj_count_prepared(
                [indexed[i], indexed[j], indexed[k]], var_ids, 3, pairs, int(threads)
            )
            slot += _gather(occurrences, nodes)
        else:
            # No sparse shortcut here: diag(A^3) needs one adjacency matrix, and the
            # three slots carry different ones. Enumerate and bincount instead.
            atoms = [
                Atom(types[i], ("a", "b"), ab),
                Atom(types[j], ("b", "c"), bc),
                Atom(types[k], ("a", "c"), ac),
            ]
            found = wcoj_join(
                atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")], backend="python"
            )
            if found.size:
                slot += _gather(np.bincount(found.ravel()), nodes)

    return columns


def typed_motif_features(
    edges_by_type: Mapping[str, np.ndarray],
    nodes: np.ndarray | None = None,
    threads: int = 0,
):
    """Per-type degree plus the typed triangle census, as one feature frame.

    ``deg__<type>`` is the free part and ``tri__<t1>_<t2>_<t3>`` is what needs the
    join. Keeping both in one frame is what makes the ablation possible: the typed
    triangles have to be shown to beat typed degree, not just untyped degree.

    Parameters
    ----------
    edges_by_type, nodes, threads
        As for :func:`typed_triangle_counts`.

    Returns
    -------
    pd.DataFrame
        Indexed by node, with one ``deg__`` column per type and one ``tri__`` column
        per sorted type triple.
    """
    import pandas as pd

    types = list(edges_by_type)
    prepared = {t: _undirected_edges(np.asarray(edges_by_type[t])) for t in types}
    if nodes is None:
        present = [e for e in prepared.values() if e.size]
        nodes = np.unique(np.concatenate(present)) if present else np.empty(0, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)

    degrees = {f"deg__{t}": _degrees(prepared[t], nodes) for t in types}
    triangles = typed_triangle_counts(prepared, nodes=nodes, threads=threads)
    return pd.concat(
        [pd.DataFrame(degrees, index=pd.Index(nodes, name="node")), triangles], axis=1
    )


# ---------------------------------------------------------------------------
# Temporal motifs
# ---------------------------------------------------------------------------
#
# Two separate things, both needed, and worth not conflating.
#
# 1. *Causality.* `flatten_relational` already filters child rows to those strictly
#    before the prediction time. The graph features did not, which is the caveat
#    recorded against the rel-event result in DESIGN.md. A cutoff fixes that.
#
# 2. *Recency and ordering.* A triangle whose three edges appeared within a week is
#    not the same feature as one that took three years, and degree cannot express
#    the difference -- which is why the temporal-motif literature (Paranjape, Benson
#    & Leskovec, WSDM 2017) reports gains that survive a degree baseline. Windows
#    give the span; phases give the order, since "two edges early, one late" is
#    triadic closure caught in the act.
#
# Phases reduce to types: a phase is an edge type that happens to be a time bucket,
# so the ordering census is `typed_triangle_counts` over phase-partitioned edges and
# needs no separate machinery.


def _as_epoch(values, what: str) -> Tuple[np.ndarray, bool]:
    """Normalise a time column to float64, reporting whether it was datetime-like."""
    import pandas as pd

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64).astype(np.float64), True
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64), False
    try:
        converted = pd.to_datetime(pd.Series(arr)).to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be numeric or datetime-like, got {arr.dtype}") from exc
    return converted.astype(np.int64).astype(np.float64), True


def _as_span(window, datetime_like: bool) -> float | None:
    """Normalise a lookback width to the same units :func:`_as_epoch` produced."""
    if window is None:
        return None
    if datetime_like:
        import pandas as pd

        return float(pd.Timedelta(window).value)
    return float(window)


def _split_phases(sub: np.ndarray, sub_t: np.ndarray, start: float, stop: float, n_phases: int):
    """Partition a window's edges into equal-*duration* phases, oldest first."""
    if len(sub) == 0:
        return {f"p{i}": np.empty((0, 2), dtype=np.int64) for i in range(n_phases)}
    inner = np.linspace(start, stop, n_phases + 1)[1:-1]
    which = np.clip(np.searchsorted(inner, sub_t, side="right"), 0, n_phases - 1)
    return {f"p{i}": sub[which == i] for i in range(n_phases)}


def temporal_motif_features(
    edges: np.ndarray,
    times,
    nodes: np.ndarray,
    cutoffs,
    windows: Mapping[str, object] | None = None,
    n_phases: int = 1,
    threads: int = 0,
):
    """Causal, windowed motif features: one row per ``(node, cutoff)`` query.

    For each query row, the graph is restricted to edges appearing **strictly before**
    that row's cutoff -- matching :func:`~tabicl.scaling.flatten_relational`, so the
    two feature families share one notion of causality -- and then, per window, to the
    edges within that lookback.

    Parameters
    ----------
    edges : np.ndarray
        Shape ``(n_edges, 2)``. Treated as undirected.

    times : array-like
        One timestamp per edge, numeric or datetime-like.

    nodes : np.ndarray
        Node id to describe, one per query row.

    cutoffs : array-like
        Prediction time for each query row, same length and time type as ``times``.

    windows : Mapping[str, object], optional
        Lookback width per named window; ``None`` as a width means all history. Widths
        are ``pd.Timedelta``-compatible for datetime times and plain numbers otherwise.
        Defaults to ``{"all": None}``.

    n_phases : int, default=1
        Split each window into this many equal-duration phases and add the ordered
        triangle census over them. ``n_phases=2`` distinguishes a triangle already
        closed early in the window (``p0_p0_p0``) from a wedge that closed late
        (``p0_p0_p1``); ``1`` skips the census entirely.

    threads : int, default=0
        Worker threads for the compiled backend; 0 means one per core.

    Returns
    -------
    pd.DataFrame
        One row per query, in input order, with a ``RangeIndex``. Columns are
        ``<window>__degree``, ``<window>__triangles``, ``<window>__clustering`` and,
        when ``n_phases > 1``, ``<window>__tri__p0_p0_p1`` and friends.

    Notes
    -----
    Cost is one motif computation per (distinct cutoff x window), times ``n_phases**3``
    joins when the census is on. Distinct cutoffs, not rows -- RelBench task tables
    have a handful of prediction timestamps shared by many rows, which is what makes
    this affordable.

    A node with no edges before its cutoff is reported as degree 0, which is
    indistinguishable from a node absent from the graph. Carry a coverage indicator
    separately if that distinction matters.

    Examples
    --------
    >>> import numpy as np
    >>> e = np.array([[0, 1], [1, 2], [0, 2]])
    >>> out = temporal_motif_features(e, [1, 2, 3], nodes=[0, 0], cutoffs=[3, 4])
    >>> out["all__triangles"].tolist()   # the closing edge lands at t=3
    [0, 1]
    """
    import pandas as pd

    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (n, 2), got {edges.shape}")
    edge_time, edges_are_dates = _as_epoch(times, "times")
    if len(edge_time) != len(edges):
        raise ValueError(f"times has length {len(edge_time)} but there are {len(edges)} edges")

    nodes = np.asarray(nodes, dtype=np.int64)
    cutoff_time, cutoffs_are_dates = _as_epoch(cutoffs, "cutoffs")
    if len(cutoff_time) != len(nodes):
        raise ValueError(
            f"cutoffs has length {len(cutoff_time)} but there are {len(nodes)} query nodes"
        )
    if edges_are_dates != cutoffs_are_dates:
        raise ValueError("times and cutoffs must both be datetime-like or both numeric")

    if windows is None:
        windows = {"all": None}
    spans = {label: _as_span(width, edges_are_dates) for label, width in windows.items()}
    if not spans:
        raise ValueError("windows is empty; pass at least one, or None for all history")
    if n_phases < 1 or n_phases > MAX_EDGE_TYPES:
        raise ValueError(f"n_phases must be between 1 and {MAX_EDGE_TYPES}, got {n_phases}")

    chronological = np.argsort(edge_time, kind="stable")
    sorted_edges, sorted_time = edges[chronological], edge_time[chronological]

    phase_combos = list(itertools.combinations_with_replacement(range(n_phases), 3))
    out: Dict[str, np.ndarray] = {}
    for label in spans:
        out[f"{label}__degree"] = np.zeros(len(nodes), dtype=np.int64)
        out[f"{label}__triangles"] = np.zeros(len(nodes), dtype=np.int64)
        out[f"{label}__clustering"] = np.zeros(len(nodes), dtype=np.float64)
        if n_phases > 1:
            for combo in phase_combos:
                name = "_".join(f"p{i}" for i in combo)
                out[f"{label}__tri__{name}"] = np.zeros(len(nodes), dtype=np.int64)

    for cutoff in np.unique(cutoff_time):
        rows = np.flatnonzero(cutoff_time == cutoff)
        asked = nodes[rows]
        # Strictly before the cutoff: an edge recorded *at* the prediction time is
        # already the future as far as that prediction is concerned.
        stop = int(np.searchsorted(sorted_time, cutoff, side="left"))

        for label, span in spans.items():
            start = 0 if span is None else int(
                np.searchsorted(sorted_time, cutoff - span, side="left")
            )
            window_edges = sorted_edges[start:stop]
            window_time = sorted_time[start:stop]

            feats = motif_features(window_edges, nodes=asked)
            out[f"{label}__degree"][rows] = feats["degree"].to_numpy()
            out[f"{label}__triangles"][rows] = feats["triangles"].to_numpy()
            out[f"{label}__clustering"][rows] = feats["clustering"].to_numpy()

            if n_phases > 1:
                if span is not None:
                    began = cutoff - span
                elif len(window_time):
                    began = float(window_time[0])
                else:
                    began = cutoff
                by_phase = _split_phases(window_edges, window_time, began, cutoff, n_phases)
                census = typed_triangle_counts(by_phase, nodes=asked, threads=threads)
                for combo in phase_combos:
                    name = "_".join(f"p{i}" for i in combo)
                    out[f"{label}__tri__{name}"][rows] = census[f"tri__{name}"].to_numpy()

    return pd.DataFrame(out, index=pd.RangeIndex(len(nodes)))


def typed_temporal_motif_features(
    edges_by_type: Mapping[str, np.ndarray],
    times_by_type: Mapping[str, object],
    nodes: np.ndarray,
    cutoffs,
    windows: Mapping[str, object] | None = None,
    threads: int = 0,
):
    """The typed census, under a per-row cutoff. Cross-type triangles included.

    Measured on rel-event, typing was worth +0.126 AUC and the cutoff cost -0.011, so
    the combination is the configuration worth having; see ``DESIGN.md``. Calling
    :func:`temporal_motif_features` once per type would not give it, because the
    cross-type columns -- ``tri__friend_coinvite_coinvite`` and the like -- are exactly
    the ones carrying the gain, and no per-type run can see them.

    Parameters
    ----------
    edges_by_type : Mapping[str, np.ndarray]
        One ``(n_edges, 2)`` array per type, over a shared node id space. At most
        :data:`MAX_EDGE_TYPES` types.

    times_by_type : Mapping[str, array-like or None]
        Timestamps for each type's edges, same keys as ``edges_by_type``. **``None``
        marks a type as static** -- every edge of it is visible at every cutoff. Real
        schemas have untimestamped relations (rel-event's `user_friends` is one) and
        dropping them costs more than including them, but a static type does leak the
        future, so the whole result is only as causal as its least causal type.

    nodes : np.ndarray
        Node id to describe, one per query row.

    cutoffs : array-like
        Prediction time per query row. Must match the time type of every non-``None``
        entry in ``times_by_type``.

    windows : Mapping[str, object], optional
        Lookback width per named window; ``None`` as a width means all history before
        the cutoff. Defaults to ``{"all": None}``. Static types ignore windows, since
        they have no time to window over.

    threads : int, default=0
        Worker threads for the compiled backend; 0 means one per core.

    Returns
    -------
    pd.DataFrame
        One row per query, in input order, with a ``RangeIndex``. Columns are
        ``<window>__deg__<type>`` and ``<window>__tri__<t1>_<t2>_<t3>``.

    Notes
    -----
    No phase splitting here, deliberately. Phases are types, so combining them would
    make the census ``(k * n_phases) ** 3`` joins -- 216 for three types and two phases
    -- and phases measured as a *negative* result on rel-event (-0.014). The untyped
    :func:`temporal_motif_features` still offers them.

    Cost is ``k ** 3`` joins per (distinct cutoff x window). Distinct cutoffs, not rows.
    """
    import pandas as pd

    types = list(edges_by_type)
    if not types:
        raise ValueError("edges_by_type is empty; give at least one edge type")
    if len(types) > MAX_EDGE_TYPES:
        raise ValueError(
            f"{len(types)} edge types needs {len(types) ** 3} joins per cutoff; "
            f"at most {MAX_EDGE_TYPES} are allowed. Merge rare types first."
        )
    missing = set(types) - set(times_by_type)
    if missing:
        raise ValueError(
            f"times_by_type has no entry for {sorted(missing)}; pass None to mark a "
            f"type static rather than omitting it"
        )

    nodes = np.asarray(nodes, dtype=np.int64)
    cutoff_time, cutoffs_are_dates = _as_epoch(cutoffs, "cutoffs")
    if len(cutoff_time) != len(nodes):
        raise ValueError(
            f"cutoffs has length {len(cutoff_time)} but there are {len(nodes)} query nodes"
        )

    # Sort each timestamped type chronologically once, so a window is a slice.
    prepared, static = {}, set()
    for name in types:
        edges = np.asarray(edges_by_type[name])
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges for type {name!r} must have shape (n, 2), got {edges.shape}")
        stamps = times_by_type[name]
        if stamps is None:
            static.add(name)
            prepared[name] = (edges, None)
            continue
        when, are_dates = _as_epoch(stamps, f"times for type {name!r}")
        if len(when) != len(edges):
            raise ValueError(
                f"times for type {name!r} has length {len(when)} but there are "
                f"{len(edges)} edges"
            )
        if are_dates != cutoffs_are_dates:
            raise ValueError(
                f"times for type {name!r} and cutoffs must both be datetime-like or "
                f"both numeric"
            )
        order = np.argsort(when, kind="stable")
        prepared[name] = (edges[order], when[order])

    if windows is None:
        windows = {"all": None}
    spans = {label: _as_span(width, cutoffs_are_dates) for label, width in windows.items()}
    if not spans:
        raise ValueError("windows is empty; pass at least one, or None for all history")

    combos = list(itertools.combinations_with_replacement(range(len(types)), 3))
    out: Dict[str, np.ndarray] = {}
    for label in spans:
        for name in types:
            out[f"{label}__deg__{name}"] = np.zeros(len(nodes), dtype=np.int64)
        for combo in combos:
            name = "_".join(types[i] for i in combo)
            out[f"{label}__tri__{name}"] = np.zeros(len(nodes), dtype=np.int64)

    # --- everything a static type contributes is loop-invariant ----------------------
    # A static type's edge set is the same at every cutoff and ignores windows, so its
    # symmetrisation, its degrees, and any triangle whose three slots are all static are
    # computed once here rather than once per prediction time. On rel-event this is the
    # difference between symmetrising a 213k-edge friendship graph twice and 46 times.
    symmetrised = {name: _undirected_edges(prepared[name][0]) for name in static}
    static_names = [n for n in types if n in static]
    static_index = {n: i for i, n in enumerate(static_names)}
    all_static = {
        tuple(sorted(static_index[types[i]] for i in combo))
        for combo in combos
        if all(types[i] in static for i in combo)
    }

    if static_names:
        # Computed over the *whole* nodes array once; per-cutoff rows index into it.
        frozen_degrees = {
            name: _degrees(symmetrised[name], nodes) for name in static_names
        }
        frozen_census = _typed_census(
            [symmetrised[n] for n in static_names], static_names, nodes, threads, all_static
        )
        for label in spans:
            for name in static_names:
                out[f"{label}__deg__{name}"] = frozen_degrees[name].copy()
            for combo, counts in frozen_census.items():
                column = "_".join(static_names[i] for i in combo)
                out[f"{label}__tri__{column}"] = counts.copy()

    # Triangles the cutoff can move: at least one slot comes from a timestamped type.
    moving = {
        combo for combo in combos if any(types[i] not in static for i in combo)
    }

    for cutoff in np.unique(cutoff_time):
        rows = np.flatnonzero(cutoff_time == cutoff)
        asked = nodes[rows]
        for label, span in spans.items():
            ordered = []
            for name in types:
                if name in static:
                    ordered.append(symmetrised[name])
                    continue
                edges, when = prepared[name]
                stop = int(np.searchsorted(when, cutoff, side="left"))
                start = 0 if span is None else int(
                    np.searchsorted(when, cutoff - span, side="left")
                )
                sub = _undirected_edges(edges[start:stop])
                ordered.append(sub)
                out[f"{label}__deg__{name}"][rows] = _degrees(sub, asked)

            census = _typed_census(ordered, types, asked, threads, moving)
            for combo in moving:
                column = "_".join(types[i] for i in combo)
                out[f"{label}__tri__{column}"][rows] = census[combo]

    return pd.DataFrame(out, index=pd.RangeIndex(len(nodes)))
