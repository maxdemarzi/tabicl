import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from tabicl import TabICLClassifier
from tabicl.scaling import (
    Table,
    chunked_set_transformer,
    collapse_kv_heads,
    expand_kv_heads,
    flatten_relational,
    kv_cache_bytes,
    row_chunked,
    think_predict_proba,
)


@pytest.fixture(scope="module")
def backbone():
    rng = np.random.RandomState(0)
    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    clf.fit(rng.rand(120, 6).astype(np.float32), rng.randint(0, 3, 120))
    return clf


# --------------------------------------------------------------------------
# 1. Row chunking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [16, 64, 10_000])
@pytest.mark.parametrize("col_chunk_size", [1, 3, 100])
@pytest.mark.parametrize("train_size", [37, None])
def test_chunked_matches_unchunked(backbone, chunk_size, col_chunk_size, train_size):
    """The whole point of the two-phase scheme is exactness, not approximation."""
    tf_col = backbone.model_.col_embedder.tf_col
    torch.manual_seed(0)
    src = torch.randn(2, 4, 91, backbone.model_.embed_dim)

    with torch.no_grad():
        expected = tf_col(src, train_size=train_size)
        got = chunked_set_transformer(
            tf_col, src, train_size, chunk_size=chunk_size, col_chunk_size=col_chunk_size
        )

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_chunked_inplace_consumes_input_but_matches(backbone):
    tf_col = backbone.model_.col_embedder.tf_col
    torch.manual_seed(1)
    src = torch.randn(1, 3, 50, backbone.model_.embed_dim)

    with torch.no_grad():
        expected = tf_col(src, train_size=20)
        got = chunked_set_transformer(tf_col, src.clone(), 20, chunk_size=16, inplace=True)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_skip_value_columns_pass_through(backbone):
    """Batched training marks padded columns with skip_value; they must survive."""
    tf_col = backbone.model_.col_embedder.tf_col
    skip = tf_col.blocks[0].skip_value
    torch.manual_seed(2)
    src = torch.randn(1, 3, 40, backbone.model_.embed_dim)
    src[0, 1] = skip

    with torch.no_grad():
        got = chunked_set_transformer(tf_col, src, 20, chunk_size=8)

    assert torch.all(got[0, 1] == skip)


def test_row_chunked_context_manager_restores_and_predicts(backbone):
    rng = np.random.RandomState(3)
    X = rng.rand(60, 6).astype(np.float32)
    original = backbone.model_.col_embedder.tf_col

    baseline = backbone.predict_proba(X)
    with row_chunked(backbone.model_, chunk_size=16, col_chunk_size=2):
        assert backbone.model_.col_embedder.tf_col is not original
        chunked = backbone.predict_proba(X)

    assert backbone.model_.col_embedder.tf_col is original
    np.testing.assert_allclose(chunked, baseline, rtol=1e-3, atol=1e-3)


def test_row_chunked_disabled_is_noop(backbone):
    original = backbone.model_.col_embedder.tf_col
    with row_chunked(backbone.model_, enabled=False):
        assert backbone.model_.col_embedder.tf_col is original


# --------------------------------------------------------------------------
# 1b. Row chunking as a first-class InferenceConfig option
# --------------------------------------------------------------------------

_CHUNK_CFG = {"COL_CONFIG": {"row_chunk": True, "row_chunk_size": 64, "col_chunk_size": 2}}


def _xy(n=260, d=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    w = rng.randn(d)
    return X, (X @ w > np.median(X @ w)).astype(int), (X @ w).astype(np.float32)


def test_config_row_chunk_matches_default_classifier():
    X, y, _ = _xy()
    base = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    base.fit(X[:200], y[:200])
    chunked = TabICLClassifier(n_estimators=1, device="cpu", random_state=0, inference_config=_CHUNK_CFG)
    chunked.fit(X[:200], y[:200])
    np.testing.assert_allclose(chunked.predict_proba(X[200:]), base.predict_proba(X[200:]), atol=1e-4)


def test_config_row_chunk_matches_default_regressor():
    from tabicl import TabICLRegressor

    X, _, y = _xy()
    base = TabICLRegressor(n_estimators=1, device="cpu", random_state=0)
    base.fit(X[:200], y[:200])
    chunked = TabICLRegressor(n_estimators=1, device="cpu", random_state=0, inference_config=_CHUNK_CFG)
    chunked.fit(X[:200], y[:200])
    np.testing.assert_allclose(chunked.predict(X[200:]), base.predict(X[200:]), atol=1e-3, rtol=1e-3)


def test_config_row_chunk_works_with_kv_cache():
    """The cached path uses a second configure() site; it must be wired too."""
    X, y, _ = _xy()
    clf = TabICLClassifier(
        n_estimators=1, device="cpu", random_state=0, kv_cache=True, inference_config=_CHUNK_CFG
    )
    clf.fit(X[:200], y[:200])
    proba = clf.predict_proba(X[200:])
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_row_chunk_defaults_off():
    from tabicl._model.inference_config import InferenceConfig

    assert InferenceConfig().COL_CONFIG.chunk_options() == {
        "row_chunk": False,
        "auto_row_chunk_threshold": 0.35,
        "row_chunk_size": 8192,
        "col_chunk_size": 32,
    }


@pytest.mark.parametrize(
    "bad, exc",
    [
        ({"row_chunk": "yes"}, ValueError),  # str is allowed ("auto"), "yes" is not
        ({"row_chunk": 1.5}, TypeError),
        ({"row_chunk_size": 0}, ValueError),
        ({"col_chunk_size": -1}, ValueError),
    ],
)
def test_row_chunk_config_validation(bad, exc):
    from tabicl._model.inference_config import InferenceConfig

    with pytest.raises(exc):
        InferenceConfig().update_from_dict({"COL_CONFIG": bad})


def test_auto_mode_validates_and_defaults():
    from tabicl._model.inference_config import InferenceConfig

    cfg = InferenceConfig()
    cfg.update_from_dict({"COL_CONFIG": {"row_chunk": "auto"}})
    opts = cfg.COL_CONFIG.chunk_options()
    assert opts["row_chunk"] == "auto"
    assert opts["auto_row_chunk_threshold"] == 0.35

    with pytest.raises(ValueError, match="row_chunk must be"):
        InferenceConfig().update_from_dict({"COL_CONFIG": {"row_chunk": "sometimes"}})
    with pytest.raises(ValueError, match="auto_row_chunk_threshold"):
        InferenceConfig().update_from_dict({"COL_CONFIG": {"auto_row_chunk_threshold": 1.5}})


@pytest.mark.parametrize(
    "mode, threshold, is_cuda, expected",
    [
        (False, 0.35, True, False),
        (True, 0.35, False, True),  # explicit True chunks even on CPU
        ("auto", 0.35, False, False),  # auto is a GPU-memory decision; CPU cannot OOM
        ("auto", 0.0, True, True),  # threshold 0 => always over budget
        ("auto", 1.0, True, False),  # tiny tensor, generous budget => skip
    ],
)
def test_chunk_wanted_decision(backbone, mode, threshold, is_cuda, expected):
    """The auto rule is a memory decision, so pin its branches without needing a GPU."""
    col = backbone.model_.col_embedder
    opts = {"row_chunk": mode, "auto_row_chunk_threshold": threshold}

    class _Fake:
        """Small stand-in so the CUDA branch is exercised without a CUDA device."""

        numel = staticmethod(lambda: 1024)
        element_size = staticmethod(lambda: 4)
        is_cuda = False
        device = "cpu"

    fake = _Fake()
    fake.is_cuda = is_cuda
    if is_cuda and mode == "auto":
        free = 1 << 30
        monkey = lambda _dev: (free, free)  # noqa: E731
        real = torch.cuda.mem_get_info
        torch.cuda.mem_get_info = monkey
        try:
            assert col._chunk_wanted(fake, opts) is expected
        finally:
            torch.cuda.mem_get_info = real
    else:
        assert col._chunk_wanted(fake, opts) is expected


def test_manager_items_excludes_chunk_keys():
    """InferenceManager.configure has no **kwargs, so these must be filtered out."""
    from tabicl._model.inference_config import InferenceConfig

    cfg = InferenceConfig().COL_CONFIG
    items = cfg.manager_items()
    assert not {"row_chunk", "row_chunk_size", "col_chunk_size"} & set(items)
    assert "offload" in items


# --------------------------------------------------------------------------
# 2. Multi-query KV cache
# --------------------------------------------------------------------------


def _fake_cache(nhead=8, src_len=32, head_dim=16, n_layers=3):
    from tabicl._model.kv_cache import KVCache, KVCacheEntry

    cache = KVCache()
    for i in range(n_layers):
        cache.kv[i] = KVCacheEntry(
            key=torch.randn(1, nhead, src_len, head_dim),
            value=torch.randn(1, nhead, src_len, head_dim),
        )
    return cache


def test_collapse_reduces_cache_by_nhead():
    cache = _fake_cache(nhead=8)
    before = kv_cache_bytes(cache)
    after = kv_cache_bytes(collapse_kv_heads(cache))
    assert before / after == pytest.approx(8.0)


def test_collapse_mean_is_the_head_mean():
    cache = _fake_cache(nhead=4)
    collapsed = collapse_kv_heads(cache, mode="mean")
    torch.testing.assert_close(collapsed.kv[0].key[:, 0], cache.kv[0].key.mean(dim=1))


def test_collapse_first_keeps_head_zero():
    cache = _fake_cache(nhead=4)
    collapsed = collapse_kv_heads(cache, mode="first")
    torch.testing.assert_close(collapsed.kv[0].key[:, 0], cache.kv[0].key[:, 0])


def test_expand_restores_head_count():
    cache = _fake_cache(nhead=8)
    expanded = expand_kv_heads(collapse_kv_heads(cache), 8)
    assert expanded.kv[0].key.shape == cache.kv[0].key.shape


def test_collapse_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        collapse_kv_heads(_fake_cache(), mode="median")


# --------------------------------------------------------------------------
# 3. Relational flattening
# --------------------------------------------------------------------------


def _toy_db():
    entity = pd.DataFrame(
        {
            "uid": [0, 1, 2],
            "age": [30, 40, 50],
            "cutoff": pd.to_datetime(["2026-01-10", "2026-01-10", "2026-01-10"]),
        }
    )
    child = pd.DataFrame(
        {
            "uid": [0, 0, 1, 1, 1],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-02", "2026-01-20", "2026-01-21"]),
            "amount": [1.0, 3.0, 10.0, 999.0, 999.0],
        }
    )
    return entity, child


def test_flatten_aggregates_and_preserves_row_order():
    entity, child = _toy_db()
    out = flatten_relational(entity, "uid", [Table(child, "uid", "txn", time_column="ts")])
    assert len(out) == len(entity)
    assert list(out["age"]) == [30, 40, 50]
    assert out["txn__count"].tolist() == [2, 3, 0]
    assert out["txn__amount__mean"].iloc[0] == pytest.approx(2.0)


def test_cutoff_excludes_future_child_rows():
    """Temporal leakage is the failure mode that matters for RelBench-style tasks."""
    entity, child = _toy_db()
    out = flatten_relational(
        entity, "uid", [Table(child, "uid", "txn", time_column="ts")], cutoff_column="cutoff"
    )
    # uid=1 has two post-cutoff rows of 999.0 that must not be aggregated.
    assert out["txn__count"].iloc[1] == 1
    assert out["txn__amount__max"].iloc[1] == pytest.approx(10.0)
    assert "cutoff" not in out.columns


def test_cutoff_requires_time_column():
    entity, child = _toy_db()
    with pytest.raises(ValueError, match="time_column"):
        flatten_relational(entity, "uid", [Table(child, "uid", "txn")], cutoff_column="cutoff")


def test_unknown_primary_key_raises():
    entity, child = _toy_db()
    with pytest.raises(ValueError, match="primary_key"):
        flatten_relational(entity, "nope", [Table(child, "uid", "txn")])


def _two_hop_db():
    """users -> orders -> items, the shape RelBench tasks actually have."""
    users = pd.DataFrame(
        {"uid": [0, 1], "age": [30, 40], "cutoff": pd.to_datetime(["2026-02-01", "2026-02-01"])}
    )
    orders = pd.DataFrame(
        {
            "oid": [10, 11, 12],
            "uid": [0, 0, 1],
            "ts": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-01-10"]),
            "total": [5.0, 7.0, 100.0],
        }
    )
    items = pd.DataFrame(
        {
            "oid": [10, 10, 11, 12],
            "ts": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-20", "2026-01-10"]),
            "price": [2.0, 3.0, 7.0, 100.0],
        }
    )
    return users, orders, items


def test_two_hop_aggregation():
    users, orders, items = _two_hop_db()
    out = flatten_relational(
        users,
        "uid",
        [
            Table(
                orders,
                foreign_key="uid",
                name="ord",
                time_column="ts",
                primary_key="oid",
                children=[Table(items, foreign_key="oid", name="item", time_column="ts")],
            )
        ],
    )
    assert len(out) == 2
    assert out["ord__count"].tolist() == [2, 1]
    # Counts roll up by ADDITION, not by averaging: user 0 owns 2 + 1 = 3 items.
    assert out["ord__item__count"].iloc[0] == pytest.approx(3.0)
    # And the mean is the true mean over all three items (2, 3, 7), not a mean of
    # per-order means -- which would be (2.5 + 7) / 2 = 4.75.
    assert out["ord__item__price__mean"].iloc[0] == pytest.approx(4.0)
    assert out["ord__item__price__max"].iloc[0] == pytest.approx(7.0)


def test_two_hop_cutoff_propagates():
    """A grandchild after the entity's cutoff is still leakage, two hops out."""
    users, orders, items = _two_hop_db()
    users = users.copy()
    users["cutoff"] = pd.to_datetime(["2026-01-10", "2026-02-01"])  # excludes order 11

    out = flatten_relational(
        users,
        "uid",
        [
            Table(
                orders,
                foreign_key="uid",
                name="ord",
                time_column="ts",
                primary_key="oid",
                children=[Table(items, foreign_key="oid", name="item", time_column="ts")],
            )
        ],
        cutoff_column="cutoff",
    )
    # user 0 keeps only order 10 (2026-01-05), so its 2 items and nothing from order 11.
    assert out["ord__count"].iloc[0] == 1
    assert out["ord__item__count"].iloc[0] == pytest.approx(2.0)
    assert out["ord__item__price__mean"].iloc[0] == pytest.approx(2.5)
    assert out["ord__total__max"].iloc[0] == pytest.approx(5.0)


def test_factorized_two_hop_equals_materialized_join():
    """The whole justification for sufficient statistics: it must equal the join.

    Aggregating level by level is only valid if the result matches aggregating the
    fully joined table. mean/std are not decomposable, so they are derived at the
    root from count/sum/sumsq, which are.
    """
    rng = np.random.default_rng(0)
    n_users, n_orders, n_items = 40, 200, 900
    users = pd.DataFrame({"uid": np.arange(n_users), "age": rng.integers(20, 70, n_users)})
    orders = pd.DataFrame({"oid": np.arange(n_orders), "uid": rng.integers(0, n_users, n_orders)})
    items = pd.DataFrame(
        {"oid": rng.integers(0, n_orders, n_items), "price": rng.gamma(2.0, 10.0, n_items)}
    )

    out = flatten_relational(
        users,
        "uid",
        [Table(orders, "uid", "ord", primary_key="oid", children=[Table(items, "oid", "item")])],
    )

    joined = items.merge(orders, on="oid").merge(users, on="uid")
    g = joined.groupby("uid")["price"]
    truth = pd.DataFrame(
        {
            "count": g.count(),
            "sum": g.sum(),
            "mean": g.mean(),
            "min": g.min(),
            "max": g.max(),
            "std": g.std(ddof=0),
        }
    ).reindex(users.uid)

    for stat in truth.columns:
        expected = truth[stat].to_numpy(dtype=float)
        actual = out[f"ord__item__price__{stat}"].to_numpy(dtype=float)
        both_nan = np.isnan(expected) & np.isnan(actual)
        np.testing.assert_allclose(expected[~both_nan], actual[~both_nan], rtol=1e-9, atol=1e-9)


def test_nested_stats_do_not_blow_up_columns():
    """Depth-k must stay linear: statistics roll up 1:1, they do not cross-multiply."""
    users = pd.DataFrame({"uid": [0, 1], "age": [1, 2]})
    orders = pd.DataFrame({"oid": [0, 1], "uid": [0, 1]})
    items = pd.DataFrame({"oid": [0, 1], "price": [1.0, 2.0]})

    one_hop = flatten_relational(users, "uid", [Table(items, "oid", "item")])
    two_hop = flatten_relational(
        users,
        "uid",
        [Table(orders, "uid", "ord", primary_key="oid", children=[Table(items, "oid", "item")])],
    )
    # The extra hop adds a bounded number of columns, not a multiplicative factor.
    assert len(two_hop.columns) < 2 * len(one_hop.columns)


def test_sumsq_is_not_leaked_as_a_feature():
    """sumsq carries variance up the tree; it is not something to train on."""
    users = pd.DataFrame({"uid": [0, 1], "age": [1, 2]})
    items = pd.DataFrame({"uid": [0, 0, 1], "price": [1.0, 3.0, 5.0]})
    out = flatten_relational(users, "uid", [Table(items, "uid", "item")])
    assert not [c for c in out.columns if c.endswith("__sumsq")]
    assert out["item__price__std"].iloc[0] == pytest.approx(1.0)


def test_children_without_primary_key_raises():
    _, orders, items = _two_hop_db()
    bad = Table(orders, "uid", "ord", time_column="ts", children=[Table(items, "oid", "item")])
    with pytest.raises(ValueError, match="primary_key"):
        flatten_relational(
            pd.DataFrame({"uid": [0, 1], "age": [1, 2]}), "uid", [bad]
        )


# --------------------------------------------------------------------------
# 4. Test-time compute
# --------------------------------------------------------------------------


def test_thinking_returns_valid_distribution():
    rng = np.random.RandomState(0)
    X = rng.rand(160, 5).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.randn(160) > 0.5).astype(int)
    est = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)

    res = think_predict_proba(est, X[:120], y[:120], X[120:], n_permutations=2, random_state=0)

    assert res.proba.shape == (40, 2)
    np.testing.assert_allclose(res.proba.sum(axis=1), 1.0, rtol=1e-5)
    assert (res.proba >= 0).all()
    assert res.blend_weight in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)


def test_thinking_never_reports_worse_than_base():
    """The validation guard must hold: 0.0 blending is always in the grid."""
    rng = np.random.RandomState(1)
    X = rng.rand(140, 4).astype(np.float32)
    y = rng.randint(0, 2, 140)  # pure noise: thinking should not claim a win
    est = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)

    res = think_predict_proba(est, X[:110], y[:110], X[110:], n_permutations=2, random_state=0)
    assert res.val_score_thought <= res.val_score_base + 1e-9


# --------------------------------------------------------------------------
# 5. Worst-case optimal joins (cyclic patterns)
# --------------------------------------------------------------------------

import itertools

from tabicl.scaling import Atom, motif_features, triangle_counts, wcoj_join


def _random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    e = np.array(pairs, dtype=np.int64)
    return np.unique(np.vstack([e, e[:, ::-1]]), axis=0)


def _brute_triangles(edges):
    present = {(int(a), int(b)) for a, b in edges}
    nodes = sorted({int(x) for x in edges.ravel()}) if edges.size else []
    return sorted(
        (a, b, c)
        for a, b, c in itertools.combinations(nodes, 3)
        if (a, b) in present and (b, c) in present and (a, c) in present
    )


@pytest.mark.parametrize("n, p, seed", [(12, 0.3, 0), (20, 0.25, 1), (15, 0.6, 2), (8, 0.9, 3)])
def test_wcoj_triangles_match_brute_force(n, p, seed):
    """A worst-case optimal join is only worth having if it is also correct."""
    edges = _random_graph(n, p, seed)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    got = wcoj_join(atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")])
    assert sorted(map(tuple, got)) == _brute_triangles(edges)


def test_symmetry_breaking_equals_filtering_afterwards():
    """Pushing a < b < c into the join must not change the answer, only the cost."""
    edges = _random_graph(30, 0.3, 7)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    unconstrained = wcoj_join(atoms, ["a", "b", "c"])
    filtered = unconstrained[
        (unconstrained[:, 0] < unconstrained[:, 1]) & (unconstrained[:, 1] < unconstrained[:, 2])
    ]
    constrained = wcoj_join(atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")])
    assert sorted(map(tuple, filtered)) == sorted(map(tuple, constrained))


def test_wcoj_four_cycle():
    """Cycles longer than a triangle are the case tree aggregation cannot express."""
    e = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3]], dtype=np.int64)
    edges = np.unique(np.vstack([e, e[:, ::-1]]), axis=0)
    present = {(int(a), int(b)) for a, b in edges}
    got = wcoj_join(
        [
            Atom("e", ("a", "b"), edges),
            Atom("e", ("b", "c"), edges),
            Atom("e", ("c", "d"), edges),
            Atom("e", ("d", "a"), edges),
        ],
        ["a", "b", "c", "d"],
    )
    brute = [
        t
        for t in itertools.product(range(1, 5), repeat=4)
        if (t[0], t[1]) in present
        and (t[1], t[2]) in present
        and (t[2], t[3]) in present
        and (t[3], t[0]) in present
    ]
    assert sorted(map(tuple, got)) == sorted(brute)


def test_wcoj_empty_and_no_match():
    empty = np.empty((0, 2), dtype=np.int64)
    assert wcoj_join([Atom("e", ("a", "b"), empty)], ["a", "b"]).shape == (0, 2)
    disjoint = np.array([[1, 2], [3, 4]], dtype=np.int64)
    out = wcoj_join(
        [Atom("e", ("a", "b"), disjoint), Atom("f", ("b", "c"), np.array([[9, 9]]))],
        ["a", "b", "c"],
    )
    assert out.shape == (0, 3)


def test_wcoj_max_results_caps_output():
    edges = _random_graph(25, 0.5, 11)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    assert len(wcoj_join(atoms, ["a", "b", "c"], max_results=5)) == 5


def test_wcoj_rejects_bad_inputs():
    e = np.array([[1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="expected"):
        Atom("e", ("a", "b", "c"), e)
    with pytest.raises(ValueError, match="not in order"):
        wcoj_join([Atom("e", ("a", "b"), e)], ["a"])
    with pytest.raises(ValueError, match="precede"):
        wcoj_join([Atom("e", ("a", "b"), e)], ["a", "b"], less_than=[("b", "a")])


def test_triangle_counts_and_motifs():
    edges = np.array([[1, 2], [2, 3], [1, 3], [3, 4]], dtype=np.int64)
    counts = triangle_counts(edges, nodes=np.array([1, 2, 3, 4]))
    assert counts.tolist() == [1, 1, 1, 0]

    feats = motif_features(edges)
    assert feats.loc[3, "degree"] == 3
    assert feats.loc[3, "triangles"] == 1
    # node 3 has 3 neighbours -> 3 wedges, 1 closed
    assert feats.loc[3, "clustering"] == pytest.approx(1 / 3)
    assert feats.loc[4, "clustering"] == 0.0


def test_native_backend_matches_python_when_built():
    """The two backends must agree; only row order may differ."""
    from tabicl.scaling import native_available

    edges = _random_graph(25, 0.4, 5)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    py_rows = wcoj_join(atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")], backend="python")
    expected = sorted(map(tuple, py_rows))
    assert expected == _brute_triangles(edges)

    if not native_available():
        pytest.skip("compiled backend not built")
    native_rows = wcoj_join(
        atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")], backend="native"
    )
    assert sorted(map(tuple, native_rows)) == expected


def test_backend_argument_is_validated():
    e = np.array([[1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="backend must be"):
        wcoj_join([Atom("e", ("a", "b"), e)], ["a", "b"], backend="fast")


def test_wcoj_count_matches_enumeration():
    """FAQ counting must agree with enumerating and counting afterwards."""
    from tabicl.scaling import native_available, wcoj_count

    if not native_available():
        pytest.skip("compiled backend not built")

    edges = _random_graph(30, 0.4, 4)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    rows = wcoj_join(atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")])
    total, occurrences = wcoj_count(atoms, ["a", "b", "c"], less_than=[("a", "b"), ("b", "c")])

    assert total == len(rows)
    expected = np.bincount(rows.ravel(), minlength=len(occurrences))
    np.testing.assert_array_equal(occurrences, expected[: len(occurrences)])


def test_wcoj_count_needs_native():
    from tabicl.scaling import native_available, wcoj_count

    if native_available():
        pytest.skip("backend is built, so the error path is unreachable")
    with pytest.raises(RuntimeError, match="compiled backend"):
        wcoj_count([Atom("e", ("a", "b"), np.array([[1, 2]]))], ["a", "b"])


# --------------------------------------------------------------------------
# 6. Semirings
# --------------------------------------------------------------------------

from tabicl.scaling import (
    BUILTIN_SEMIRINGS,
    MAX_PLUS,
    MIN_PLUS,
    SUM_PRODUCT,
    Semiring,
    check_semiring_laws,
    hop_product,
)


@pytest.mark.parametrize("name", sorted(BUILTIN_SEMIRINGS))
def test_builtin_semirings_satisfy_the_axioms(name):
    ring = BUILTIN_SEMIRINGS[name]
    samples = [False, True] if name == "boolean" else [0.0, 1.0, 2.5, -3.0]
    check_semiring_laws(ring, samples)


def test_check_semiring_laws_rejects_a_broken_one():
    """A structure that fails distributivity still produces numbers -- silently wrong."""
    broken = Semiring(name="broken", zero=0.0, one=1.0, add=lambda x, y: x + y, mul=lambda x, y: x + y)
    with pytest.raises(ValueError, match="distribute|identity|annihilate"):
        check_semiring_laws(broken, [1.0, 2.0, 3.0])


def test_invertibility_matches_maintainability():
    """Only invertible rings can retract a deleted row from a maintained aggregate."""
    assert SUM_PRODUCT.invertible
    assert not MIN_PLUS.invertible
    assert not MAX_PLUS.invertible


def test_semiring_fold_helpers():
    assert SUM_PRODUCT.sum([1.0, 2.0, 3.0]) == 6.0
    assert SUM_PRODUCT.product([2.0, 3.0]) == 6.0
    assert SUM_PRODUCT.sum([]) == SUM_PRODUCT.zero
    assert MIN_PLUS.sum([3.0, 1.0, 2.0]) == 1.0
    assert MIN_PLUS.product([3.0, 1.0]) == 4.0  # mul is addition


def test_hop_product_combines_across_a_hop():
    """The operation the roll-up could not express: mul across a hop, not add."""
    users = pd.DataFrame({"uid": [0, 1], "age": [30, 40]})
    orders = pd.DataFrame({"oid": [10, 11, 12], "uid": [0, 0, 1], "discount": [0.5, 0.25, 1.0]})
    items = pd.DataFrame({"oid": [10, 10, 11, 12], "price": [2.0, 3.0, 7.0, 100.0]})

    feat = flatten_relational(
        users,
        "uid",
        [Table(orders, "uid", "ord", primary_key="oid", children=[Table(items, "oid", "item")])],
    )
    value = hop_product(feat, "ord__discount__mean", "ord__item__price__sum", "v")
    # user 0: mean discount (0.5 + 0.25)/2 = 0.375, item total 2 + 3 + 7 = 12
    assert value.iloc[0] == pytest.approx(0.375 * 12.0)
    # tropical mul is addition, so the same hop accumulates instead of scaling
    tropical = hop_product(feat, "ord__discount__mean", "ord__item__price__sum", "v", semiring=MIN_PLUS)
    assert tropical.iloc[0] == pytest.approx(0.375 + 12.0)


def test_hop_product_rejects_missing_column():
    frame = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="not in frame"):
        hop_product(frame, "a", "nope", "x")


# --------------------------------------------------------------------------
# 7. FAQ aggregation over a semiring
# --------------------------------------------------------------------------


def _triangles_of(edges):
    present = {(int(a), int(b)) for a, b in edges}
    nodes = sorted({int(x) for x in edges.ravel()})
    return [
        (a, b, c)
        for a, b, c in itertools.combinations(nodes, 3)
        if (a, b) in present and (b, c) in present and (a, c) in present
    ]


@pytest.mark.parametrize("ring", [SUM_PRODUCT, MIN_PLUS, MAX_PLUS])
def test_wcoj_aggregate_matches_brute_force(ring):
    """mul accumulates along a witness, add combines witnesses -- verified directly."""
    from tabicl.scaling import native_available, wcoj_aggregate

    if not native_available():
        pytest.skip("compiled backend not built")

    edges = _random_graph(18, 0.45, 9)
    atoms = [
        Atom("e", ("a", "b"), edges),
        Atom("e", ("b", "c"), edges),
        Atom("e", ("a", "c"), edges),
    ]
    rng = np.random.default_rng(3)
    weights = rng.uniform(1.0, 5.0, int(edges.max()) + 1)

    total, overall, per_value = wcoj_aggregate(
        atoms, ["a", "b", "c"], weights, semiring=ring, less_than=[("a", "b"), ("b", "c")]
    )
    triangles = _triangles_of(edges)
    assert total == len(triangles)

    def payload(t):
        vals = [weights[v] for v in t]
        return float(np.prod(vals)) if ring is SUM_PRODUCT else float(np.sum(vals))

    reduce = {SUM_PRODUCT: sum, MIN_PLUS: min, MAX_PLUS: max}[ring]
    assert overall == pytest.approx(reduce([payload(t) for t in triangles]))

    for node in {v for t in triangles for v in t}:
        expected = reduce([payload(t) for t in triangles if node in t])
        assert per_value[node] == pytest.approx(expected)


def test_wcoj_aggregate_rejects_unsupported_semiring():
    from tabicl.scaling import BOOLEAN, native_available, wcoj_aggregate

    if not native_available():
        pytest.skip("compiled backend not built")
    e = np.array([[1, 2], [2, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="not supported by the compiled kernel"):
        wcoj_aggregate([Atom("e", ("a", "b"), e)], ["a", "b"], np.ones(3), semiring=BOOLEAN)


def test_wcoj_aggregate_rejects_short_weights():
    from tabicl.scaling import native_available, wcoj_aggregate

    if not native_available():
        pytest.skip("compiled backend not built")
    e = np.array([[1, 5], [5, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="weights has length"):
        wcoj_aggregate([Atom("e", ("a", "b"), e)], ["a", "b"], np.ones(2))


def test_repeated_entity_keys_get_per_row_features():
    """The standard predictive-relational shape: one row per (entity, prediction time).

    RelBench's rel-f1 averages ~15 rows per driver and reaches 59. Grouping by key
    would collapse those into one set of features, and the cutoff would be ambiguous
    besides, so aggregation is per entity *row*.
    """
    entity = pd.DataFrame(
        {
            "uid": [0, 0, 1],
            "cutoff": pd.to_datetime(["2026-01-10", "2026-02-10", "2026-02-10"]),
        }
    )
    child = pd.DataFrame(
        {
            "uid": [0, 0, 1],
            "ts": pd.to_datetime(["2026-01-05", "2026-02-05", "2026-01-05"]),
            "amount": [1.0, 100.0, 7.0],
        }
    )

    out = flatten_relational(
        entity, "uid", [Table(child, "uid", "ev", time_column="ts")], cutoff_column="cutoff"
    )

    assert len(out) == 3
    # Same driver, two different cutoffs -> different history, so different features.
    assert out["ev__count"].tolist() == [1, 2, 1]
    assert out["ev__amount__sum"].iloc[0] == pytest.approx(1.0)
    assert out["ev__amount__sum"].iloc[1] == pytest.approx(101.0)
    assert out["ev__amount__sum"].iloc[2] == pytest.approx(7.0)


def test_entity_rows_without_history_get_zero_count_not_zero_sum():
    """No history means a count of 0 but an *unknown* mean -- not 0."""
    entity = pd.DataFrame({"uid": [0, 1], "cutoff": pd.to_datetime(["2026-01-10"] * 2)})
    child = pd.DataFrame({"uid": [0], "ts": pd.to_datetime(["2026-01-05"]), "amount": [4.0]})

    out = flatten_relational(
        entity, "uid", [Table(child, "uid", "ev", time_column="ts")], cutoff_column="cutoff"
    )
    assert out["ev__count"].tolist() == [1, 0]
    assert np.isnan(out["ev__amount__mean"].iloc[1])


def test_repeated_keys_with_nested_children_is_rejected():
    """A grandchild's cutoff is ambiguous when the entity key repeats; say so."""
    entity = pd.DataFrame({"uid": [0, 0], "cutoff": pd.to_datetime(["2026-01-10"] * 2)})
    orders = pd.DataFrame({"oid": [1], "uid": [0], "ts": pd.to_datetime(["2026-01-05"])})
    items = pd.DataFrame({"oid": [1], "price": [2.0]})

    nested = Table(orders, "uid", "ord", time_column="ts", primary_key="oid",
                   children=[Table(items, "oid", "item")])
    with pytest.raises(ValueError, match="entity keys repeat"):
        flatten_relational(entity, "uid", [nested], cutoff_column="cutoff")


def test_negative_values_are_rejected_not_written_out_of_bounds():
    """pd.factorize emits -1 for nulls, and the kernels index arrays *by value*."""
    from tabicl.scaling import native_available, wcoj_count

    if not native_available():
        pytest.skip("compiled backend not built")
    bad = np.array([[0, 1], [1, -1]], dtype=np.int64)
    with pytest.raises(ValueError, match="non-negative"):
        wcoj_count([Atom("e", ("a", "b"), bad)], ["a", "b"])


# --------------------------------------------------------------------------
# 6. Typed and temporal motifs
# --------------------------------------------------------------------------

from tabicl.scaling import (
    temporal_motif_features,
    typed_motif_features,
    typed_triangle_counts,
)


def _typed_random_graph(n, p, n_types, seed):
    """Undirected graph with one type per edge, as {type: (m, 2) array}."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    labels = rng.integers(0, n_types, len(pairs))
    out = {}
    for t in range(n_types):
        chosen = [pair for pair, lab in zip(pairs, labels) if lab == t]
        out[f"t{t}"] = (
            np.array(chosen, dtype=np.int64) if chosen else np.empty((0, 2), dtype=np.int64)
        )
    return out


def _brute_typed_triangles(edges_by_type, nodes):
    """Enumerate node triples directly; no join, no symmetry breaking."""
    types = list(edges_by_type)
    present = {}
    for t in types:
        e = np.asarray(edges_by_type[t])
        present[t] = {(int(a), int(b)) for a, b in e} | {(int(b), int(a)) for a, b in e}

    counts = {
        combo: {int(v): 0 for v in nodes}
        for combo in itertools.combinations_with_replacement(types, 3)
    }
    for a, b, c in itertools.combinations(sorted(int(v) for v in nodes), 3):
        for tab in (t for t in types if (a, b) in present[t]):
            for tbc in (t for t in types if (b, c) in present[t]):
                for tac in (t for t in types if (a, c) in present[t]):
                    key = tuple(sorted((tab, tbc, tac), key=types.index))
                    for node in (a, b, c):
                        counts[key][node] += 1
    return counts


@pytest.mark.parametrize("n, p, n_types, seed", [(14, 0.4, 2, 0), (12, 0.5, 3, 1), (16, 0.3, 3, 2)])
def test_typed_triangles_match_brute_force(n, p, n_types, seed):
    """The type-per-slot argument is the whole correctness claim; check it directly."""
    by_type = _typed_random_graph(n, p, n_types, seed)
    nodes = np.arange(n, dtype=np.int64)
    got = typed_triangle_counts(by_type, nodes=nodes)
    expected = _brute_typed_triangles(by_type, nodes)

    for combo, per_node in expected.items():
        column = "tri__" + "_".join(combo)
        assert column in got.columns
        assert got[column].tolist() == [per_node[int(v)] for v in nodes], column


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_typed_triangles_partition_the_untyped_count(seed):
    """Types are disjoint here, so the census must sum back to the plain count.

    This is the property that makes type splitting a refinement rather than a
    different feature: no triangle is dropped and none is counted twice.
    """
    by_type = _typed_random_graph(15, 0.4, 3, seed)
    nodes = np.arange(15, dtype=np.int64)
    union = np.vstack([e for e in by_type.values() if e.size])

    census = typed_triangle_counts(by_type, nodes=nodes)
    assert census.sum(axis=1).tolist() == triangle_counts(union, nodes=nodes).tolist()


def test_typed_triangles_distinguish_what_degree_cannot():
    """Two nodes with identical degree and identical triangle count, different types.

    If this passed with untyped features the split would be pointless, so the test
    asserts the untyped columns really do collide first.
    """
    # 0-1-2 all-friend triangle; 3-4-5 all-colleague triangle.
    friend = np.array([[0, 1], [1, 2], [0, 2], [3, 4]], dtype=np.int64)
    colleague = np.array([[3, 5], [4, 5]], dtype=np.int64)
    nodes = np.array([0, 3], dtype=np.int64)

    plain = motif_features(np.vstack([friend, colleague]), nodes=nodes)
    assert plain["degree"].tolist() == [2, 2]
    assert plain["triangles"].tolist() == [1, 1]  # indistinguishable

    typed = typed_motif_features({"f": friend, "c": colleague}, nodes=nodes)
    assert typed.loc[0, "tri__f_f_f"] == 1
    assert typed.loc[0, "tri__f_c_c"] == 0
    assert typed.loc[3, "tri__f_f_f"] == 0
    assert typed.loc[3, "tri__f_c_c"] == 1
    assert typed.loc[3, "deg__f"] == 1 and typed.loc[3, "deg__c"] == 1


def test_typed_motifs_handle_empty_types_and_reject_too_many():
    empty = np.empty((0, 2), dtype=np.int64)
    out = typed_triangle_counts({"a": np.array([[0, 1], [1, 2], [0, 2]]), "b": empty})
    assert out["tri__a_a_a"].loc[0] == 1
    assert out["tri__a_a_b"].sum() == 0

    with pytest.raises(ValueError, match="at least one edge type"):
        typed_triangle_counts({})
    with pytest.raises(ValueError, match="at most 6"):
        typed_triangle_counts({f"t{i}": empty for i in range(7)})


def test_temporal_features_are_strictly_causal():
    """The caveat this exists to remove: an edge at the cutoff is still the future."""
    e = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    out = temporal_motif_features(e, [1, 2, 3], nodes=[0, 0, 0], cutoffs=[3, 4, 10])

    # At cutoff 3 the closing edge has not happened yet -- a wedge, not a triangle.
    assert out["all__degree"].tolist() == [1, 2, 2]
    assert out["all__triangles"].tolist() == [0, 1, 1]


def test_temporal_features_match_static_motifs_once_all_history_is_in():
    """With every edge before the cutoff and no window, this must reduce to the static case."""
    edges = _random_graph(20, 0.3, 4)
    times = np.arange(len(edges))
    nodes = np.arange(20, dtype=np.int64)

    static = motif_features(edges, nodes=nodes)
    temporal = temporal_motif_features(
        edges, times, nodes=nodes, cutoffs=np.full(len(nodes), len(edges) + 1)
    )
    assert temporal["all__degree"].tolist() == static["degree"].tolist()
    assert temporal["all__triangles"].tolist() == static["triangles"].tolist()
    np.testing.assert_allclose(temporal["all__clustering"], static["clustering"].to_numpy())


def test_temporal_window_drops_stale_edges():
    """A window is the part degree cannot fake: same node, same total degree, older ties."""
    # Node 0 closes a triangle long ago and gains two fresh unrelated neighbours.
    e = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    t = [0, 0, 0, 100, 100]
    out = temporal_motif_features(
        e, t, nodes=[0, 0], cutoffs=[101, 101], windows={"all": None, "recent": 10}
    )
    assert out["all__degree"].iloc[0] == 4 and out["all__triangles"].iloc[0] == 1
    assert out["recent__degree"].iloc[0] == 2 and out["recent__triangles"].iloc[0] == 0


def test_temporal_phases_separate_closure_in_time_from_old_triangles():
    """Ordering is the point: a wedge that closes late is not an already-closed triangle."""
    # 0-1-2 fully formed early; 3-4-5 is a wedge early that closes in the late phase.
    e = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int64)
    t = [0, 1, 2, 0, 1, 9]
    out = temporal_motif_features(
        e, t, nodes=[0, 3], cutoffs=[10, 10], windows={"w": 10}, n_phases=2
    )
    # Window is [0, 10), so phase 0 is [0, 5) and phase 1 is [5, 10).
    assert out["w__tri__p0_p0_p0"].tolist() == [1, 0]
    assert out["w__tri__p0_p0_p1"].tolist() == [0, 1]
    # Both nodes look identical without the ordering.
    assert out["w__triangles"].tolist() == [1, 1]


@pytest.mark.parametrize("n_phases", [2, 3])
def test_temporal_phase_census_sums_to_the_window_count(n_phases):
    """Phases are types, so the same partition property has to hold."""
    edges = _random_graph(16, 0.35, 9)
    rng = np.random.default_rng(3)
    times = rng.integers(0, 50, len(edges))
    # Both directions of an edge must share a timestamp or the phases disagree.
    keyed = {}
    for (a, b), t in zip(edges, times):
        keyed.setdefault((min(a, b), max(a, b)), int(t))
    times = np.array([keyed[(min(a, b), max(a, b))] for a, b in edges])

    nodes = np.arange(16, dtype=np.int64)
    out = temporal_motif_features(
        edges, times, nodes=nodes, cutoffs=np.full(len(nodes), 60), n_phases=n_phases
    )
    census = out[[c for c in out.columns if "__tri__" in c]]
    assert census.sum(axis=1).tolist() == out["all__triangles"].tolist()


def test_temporal_features_accept_datetimes():
    """RelBench task tables carry pandas timestamps, not integers."""
    e = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    t = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-03-01"])
    out = temporal_motif_features(
        e,
        t,
        nodes=[0, 0],
        cutoffs=pd.to_datetime(["2026-02-01", "2026-04-01"]),
        windows={"30d": pd.Timedelta("30D")},
    )
    assert out["30d__triangles"].tolist() == [0, 0]  # the 30d window never spans all three
    full = temporal_motif_features(
        e, t, nodes=[0, 0], cutoffs=pd.to_datetime(["2026-02-01", "2026-04-01"])
    )
    assert full["all__triangles"].tolist() == [0, 1]


def test_typed_temporal_matches_typed_motifs_on_the_edges_it_can_see():
    """Under a cutoff past everything, the causal census must equal the static one."""
    from tabicl.scaling import typed_temporal_motif_features

    a = _random_graph(14, 0.35, 21)
    b = _random_graph(14, 0.35, 22)
    nodes = np.arange(14, dtype=np.int64)
    ta, tb = np.arange(len(a)), np.arange(len(b))

    static = typed_motif_features({"a": a, "b": b}, nodes=nodes)
    causal = typed_temporal_motif_features(
        {"a": a, "b": b},
        {"a": ta, "b": tb},
        nodes=nodes,
        cutoffs=np.full(len(nodes), max(len(a), len(b)) + 1),
    )
    for col in static.columns:
        assert causal[f"all__{col}"].tolist() == static[col].tolist(), col


def test_typed_temporal_keeps_cross_type_triangles_a_per_type_run_would_lose():
    """The whole reason this exists: one call per type cannot see a mixed triangle."""
    from tabicl.scaling import typed_temporal_motif_features

    friend = np.array([[0, 1]], dtype=np.int64)
    coinvite = np.array([[1, 2], [0, 2]], dtype=np.int64)
    out = typed_temporal_motif_features(
        {"friend": friend, "coinvite": coinvite},
        {"friend": [0], "coinvite": [1, 2]},
        nodes=[0, 0],
        cutoffs=[2, 3],
    )
    # At cutoff 2 the closing co-invite edge has not happened; at 3 it has.
    assert out["all__tri__friend_coinvite_coinvite"].tolist() == [0, 1]
    assert out["all__tri__friend_friend_friend"].tolist() == [0, 0]


def test_typed_temporal_static_types_ignore_the_cutoff():
    """A None timestamp means 'always visible' -- and that is a documented leak."""
    from tabicl.scaling import typed_temporal_motif_features

    friend = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    coinvite = np.array([[0, 3]], dtype=np.int64)
    out = typed_temporal_motif_features(
        {"friend": friend, "coinvite": coinvite},
        {"friend": None, "coinvite": [99]},
        nodes=[0, 0],
        cutoffs=[0, 100],
    )
    # The friend triangle is visible even at cutoff 0, because friend is static.
    assert out["all__tri__friend_friend_friend"].tolist() == [1, 1]
    # The co-invite edge at t=99 is not, at cutoff 0.
    assert out["all__deg__coinvite"].tolist() == [0, 1]


def test_typed_temporal_windows_apply_per_type():
    from tabicl.scaling import typed_temporal_motif_features

    a = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
    out = typed_temporal_motif_features(
        {"a": a}, {"a": [0, 90, 95]}, nodes=[0], cutoffs=[100],
        windows={"all": None, "recent": 20},
    )
    assert out["all__deg__a"].iloc[0] == 3
    assert out["recent__deg__a"].iloc[0] == 2


def test_typed_temporal_caching_matches_the_naive_per_cutoff_path():
    """Static types are hoisted out of the cutoff loop; that must be invisible.

    The reference here is the obvious implementation -- slice every type at each cutoff
    and call the static census -- which is exactly what the optimisation replaces.
    """
    from tabicl.scaling import typed_temporal_motif_features

    rng = np.random.default_rng(5)
    friend = _random_graph(18, 0.3, 31)
    invite = _random_graph(18, 0.3, 32)
    attend = _random_graph(18, 0.25, 33)
    # Both directions of an edge must share a timestamp or the slices disagree.
    def stamp(edges, seed):
        r = np.random.default_rng(seed)
        keyed = {}
        for a, b in edges:
            keyed.setdefault((min(a, b), max(a, b)), int(r.integers(0, 50)))
        return np.array([keyed[(min(a, b), max(a, b))] for a, b in edges])

    t_inv, t_att = stamp(invite, 1), stamp(attend, 2)
    nodes = rng.integers(0, 18, 40).astype(np.int64)
    cutoffs = rng.choice([10, 25, 40, 60], size=40)

    got = typed_temporal_motif_features(
        {"friend": friend, "invite": invite, "attend": attend},
        {"friend": None, "invite": t_inv, "attend": t_att},
        nodes=nodes,
        cutoffs=cutoffs,
    )

    for row in range(len(nodes)):
        cut = cutoffs[row]
        reference = typed_motif_features(
            {
                "friend": friend,  # static: the cutoff does not apply
                "invite": invite[t_inv < cut],
                "attend": attend[t_att < cut],
            },
            nodes=np.array([nodes[row]], dtype=np.int64),
        )
        for col in reference.columns:
            assert got[f"all__{col}"].iloc[row] == reference[col].iloc[0], (row, col)


def test_typed_temporal_rejects_missing_or_mismatched_times():
    from tabicl.scaling import typed_temporal_motif_features

    e = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="pass None to mark a type static"):
        typed_temporal_motif_features({"a": e, "b": e}, {"a": [0]}, nodes=[0], cutoffs=[1])
    with pytest.raises(ValueError, match="has length"):
        typed_temporal_motif_features({"a": e}, {"a": [0, 1]}, nodes=[0], cutoffs=[1])
    with pytest.raises(ValueError, match="datetime-like or"):
        typed_temporal_motif_features(
            {"a": e}, {"a": [0]}, nodes=[0], cutoffs=pd.to_datetime(["2026-01-01"])
        )


def test_temporal_features_reject_mismatched_inputs():
    e = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="times has length"):
        temporal_motif_features(e, [1], nodes=[0], cutoffs=[5])
    with pytest.raises(ValueError, match="cutoffs has length"):
        temporal_motif_features(e, [1, 2], nodes=[0, 1], cutoffs=[5])
    with pytest.raises(ValueError, match="datetime-like or both numeric"):
        temporal_motif_features(
            e, [1, 2], nodes=[0], cutoffs=pd.to_datetime(["2026-01-01"])
        )
    with pytest.raises(ValueError, match="n_phases must be"):
        temporal_motif_features(e, [1, 2], nodes=[0], cutoffs=[5], n_phases=0)


def test_chunked_icl_encoder_matches_unchunked(backbone):
    """The ICL stack is chunkable over query rows: only train rows supply K/V."""
    from tabicl.scaling._rowchunk import chunked_icl_encoder

    tf_icl = backbone.model_.icl_predictor.tf_icl
    d_model = tf_icl.blocks[0].linear1.in_features
    torch.manual_seed(0)
    src = torch.randn(1, 130, d_model)

    with torch.no_grad():
        expected = tf_icl(src, train_size=60)
        for chunk in (16, 64, 10_000):
            got = chunked_icl_encoder(tf_icl, src, 60, chunk_size=chunk)
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_chunked_icl_encoder_refuses_rope():
    """A chunk would be encoded at the wrong absolute positions, so refuse."""
    from tabicl.scaling._rowchunk import chunked_icl_encoder

    class _WithRope:
        rope = object()
        blocks = []

    with pytest.raises(ValueError, match="rope"):
        chunked_icl_encoder(_WithRope(), torch.zeros(1, 4, 8), 2)


def test_icl_chunking_via_config_preserves_labels():
    X, y, _ = _xy(n=300, d=8)
    base = TabICLClassifier(n_estimators=1, device="cpu", random_state=0)
    base.fit(X[:200], y[:200])
    chunked = TabICLClassifier(
        n_estimators=1,
        device="cpu",
        random_state=0,
        inference_config={"ICL_CONFIG": {"row_chunk": True, "row_chunk_size": 32}},
    )
    chunked.fit(X[:200], y[:200])
    p_base, p_chunk = base.predict_proba(X[200:]), chunked.predict_proba(X[200:])
    np.testing.assert_array_equal(p_chunk.argmax(1), p_base.argmax(1))


def test_windows_restrict_to_recent_history():
    """A window sees only [cutoff - window, cutoff); all-history still sees everything."""
    entity = pd.DataFrame({"uid": [0, 1], "cutoff": pd.to_datetime(["2026-03-01"] * 2)})
    child = pd.DataFrame(
        {
            "uid": [0, 0, 0, 1],
            "ts": pd.to_datetime(["2026-02-25", "2026-01-01", "2025-06-01", "2026-02-20"]),
            "amount": [10.0, 100.0, 1000.0, 5.0],
        }
    )
    out = flatten_relational(
        entity,
        "uid",
        [Table(child, "uid", "ev", time_column="ts",
               windows=[pd.Timedelta(days=30), pd.Timedelta(days=90)])],
        cutoff_column="cutoff",
    )
    assert out["ev__count"].iloc[0] == 3
    assert out["ev_30d__count"].iloc[0] == 1
    assert out["ev_90d__count"].iloc[0] == 2
    # The all-time mean dilutes recency; the windows are what expose it.
    assert out["ev__amount__mean"].iloc[0] == pytest.approx(370.0)
    assert out["ev_90d__amount__mean"].iloc[0] == pytest.approx(55.0)
    assert out["ev_30d__amount__mean"].iloc[0] == pytest.approx(10.0)


def test_windows_without_a_cutoff_are_rejected():
    entity = pd.DataFrame({"uid": [0]})
    child = pd.DataFrame({"uid": [0], "ts": pd.to_datetime(["2026-01-01"]), "v": [1.0]})
    with pytest.raises(ValueError, match="windows"):
        flatten_relational(
            entity, "uid",
            [Table(child, "uid", "ev", time_column="ts", windows=[pd.Timedelta(days=7)])],
        )


def test_all_null_categorical_group_has_no_mode():
    """value_counts() drops NaN, so a non-empty all-null group has no mode."""
    entity = pd.DataFrame({"uid": [0], "cutoff": pd.to_datetime(["2026-02-01"])})
    child = pd.DataFrame(
        {"uid": [0, 0], "ts": pd.to_datetime(["2026-01-01"] * 2), "label": [None, None]}
    )
    out = flatten_relational(
        entity, "uid", [Table(child, "uid", "ev", time_column="ts")], cutoff_column="cutoff"
    )
    assert out["ev__count"].iloc[0] == 2
    assert pd.isna(out["ev__label__mode"].iloc[0])


def test_duplicate_child_names_are_rejected():
    """Colliding names produce duplicate columns; pandas then fails opaquely."""
    entity = pd.DataFrame({"uid": [0]})
    child = pd.DataFrame({"uid": [0], "v": [1.0]})
    with pytest.raises(ValueError, match="unique"):
        flatten_relational(
            entity, "uid", [Table(child, "uid", "ev"), Table(child, "uid", "ev")]
        )


def test_asof_matches_join_path_on_invertible_statistics():
    """Same answers, different algorithm: prefix differences instead of a join."""
    from tabicl.scaling import asof_statistics

    rng = np.random.default_rng(0)
    ent = pd.DataFrame({
        "uid": rng.integers(0, 40, 200),
        "cut": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, 200), unit="D"),
    })
    ch = pd.DataFrame({
        "uid": rng.integers(0, 40, 5000),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, 5000), unit="D"),
        "amt": rng.normal(size=5000),
    })
    tbl = Table(ch, "uid", "ev", time_column="ts", windows=[pd.Timedelta(days=30)])
    joined = flatten_relational(ent, "uid", [tbl], cutoff_column="cut")
    scanned = asof_statistics(tbl, ent["uid"].to_numpy(), ent["cut"].to_numpy())

    for col in ("ev__count", "ev__amt__mean", "ev_30d__count", "ev_30d__amt__mean"):
        a = joined[col].to_numpy(dtype=float)
        b = scanned[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-9)


def test_asof_std_survives_large_offsets():
    """The join path loses every digit here; the shifted accumulator does not."""
    from tabicl.scaling import asof_statistics

    rng = np.random.default_rng(1)
    ent = pd.DataFrame({"uid": [0] * 5,
                        "cut": pd.to_datetime(["2026-06-01"] * 5)})
    ch = pd.DataFrame({
        "uid": [0] * 400,
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(400), unit="D"),
        "amt": rng.normal(size=400) + 1e9,
    })
    tbl = Table(ch, "uid", "ev", time_column="ts")
    scanned = asof_statistics(tbl, ent["uid"].to_numpy(), ent["cut"].to_numpy())
    truth = ch.loc[ch["ts"] < pd.Timestamp("2026-06-01"), "amt"].std(ddof=0)
    assert scanned["ev__amt__std"].iloc[0] == pytest.approx(truth, abs=1e-6)


def test_asof_requires_a_time_column():
    from tabicl.scaling import asof_statistics

    ch = pd.DataFrame({"uid": [0], "amt": [1.0]})
    with pytest.raises(ValueError, match="time_column"):
        asof_statistics(Table(ch, "uid", "ev"), np.array([0]), np.array([0]))


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9])
def test_std_survives_large_magnitude_columns(offset):
    """sumsq of raw values then E[X^2]-E[X]^2 returned std 18.5 for a true 1.0 at 1e9."""
    rng = np.random.default_rng(0)
    ent = pd.DataFrame({"uid": [0] * 3, "cut": pd.to_datetime(["2026-06-01"] * 3)})
    ch = pd.DataFrame({
        "uid": [0] * 400,
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(400), unit="D"),
        "amt": rng.normal(size=400) + offset,
    })
    out = flatten_relational(
        ent, "uid", [Table(ch, "uid", "ev", time_column="ts")], cutoff_column="cut"
    )
    seen = ch.loc[ch["ts"] < pd.Timestamp("2026-06-01"), "amt"]
    assert out["ev__amt__std"].iloc[0] == pytest.approx(seen.std(ddof=0), abs=1e-8)
    assert out["ev__amt__mean"].iloc[0] == pytest.approx(seen.mean(), rel=1e-12)
    assert out["ev__amt__sum"].iloc[0] == pytest.approx(seen.sum(), rel=1e-12)


def test_pivot_carriers_do_not_leak_into_features():
    """sumsq and shift are carriers; a model should never see them."""
    ent = pd.DataFrame({"uid": [0], "cut": pd.to_datetime(["2026-02-01"])})
    ch = pd.DataFrame({"uid": [0, 0], "ts": pd.to_datetime(["2026-01-01"] * 2), "v": [1.0, 3.0]})
    out = flatten_relational(
        ent, "uid", [Table(ch, "uid", "ev", time_column="ts")], cutoff_column="cut"
    )
    assert not [c for c in out.columns if c.endswith("__sumsq") or c.endswith("__shift")]
    assert out["ev__v__std"].iloc[0] == pytest.approx(1.0)


def test_asof_min_max_match_the_join_path():
    """min/max are not invertible, so they are computed per range, not by difference."""
    from tabicl.scaling import asof_statistics

    rng = np.random.default_rng(2)
    ent = pd.DataFrame({
        "uid": rng.integers(0, 30, 150),
        "cut": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, 150), unit="D"),
    })
    ch = pd.DataFrame({
        "uid": rng.integers(0, 30, 4000),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, 4000), unit="D"),
        "amt": rng.normal(size=4000),
    })
    tbl = Table(ch, "uid", "ev", time_column="ts", windows=[pd.Timedelta(days=30)])
    joined = flatten_relational(ent, "uid", [tbl], cutoff_column="cut")
    scanned = asof_statistics(tbl, ent["uid"].to_numpy(), ent["cut"].to_numpy())

    for col in ("ev__amt__min", "ev__amt__max", "ev_30d__amt__min", "ev_30d__amt__max"):
        a = joined[col].to_numpy(dtype=float)
        b = scanned[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-9)


def test_asof_prefix_nunique_matches_join_path():
    """Distinct-count over a key's prefix is a cumulative sum of first occurrences."""
    from tabicl.scaling import asof_statistics

    rng = np.random.default_rng(5)
    ent = pd.DataFrame({
        "uid": rng.integers(0, 20, 120),
        "cut": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, 120), unit="D"),
    })
    ch = pd.DataFrame({
        "uid": rng.integers(0, 20, 3000),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, 3000), unit="D"),
        "cat": rng.integers(0, 9, 3000).astype(str),
    })
    tbl = Table(ch, "uid", "ev", time_column="ts")
    joined = flatten_relational(ent, "uid", [tbl], cutoff_column="cut")
    scanned = asof_statistics(tbl, ent["uid"].to_numpy(), ent["cut"].to_numpy())

    a = np.nan_to_num(joined["ev__cat__nunique"].to_numpy(dtype=float))
    b = scanned["ev__cat__nunique"].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b)


def test_asof_does_not_claim_to_provide_mode():
    """mode has no linear range algorithm; it must stay on the join path."""
    from tabicl.scaling import asof_statistics

    ch = pd.DataFrame({
        "uid": [0, 0], "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]), "cat": ["a", "b"],
    })
    scanned = asof_statistics(
        Table(ch, "uid", "ev", time_column="ts"),
        np.array([0]), np.array([np.datetime64("2026-02-01")]),
    )
    assert not [c for c in scanned.columns if c.endswith("__mode")]
    assert "ev__cat__nunique" in scanned.columns


def test_prune_drops_only_the_genuinely_empty_by_default():
    """Defaults are conservative: aggressive pruning measured -0.095 AUC on rel-event."""
    from tabicl.scaling import prune_features

    df = pd.DataFrame({
        "good": np.arange(100.0),
        "constant": np.ones(100),
        "half_missing": [1.0, 2.0] * 25 + [np.nan] * 50,
        "all_missing": [np.nan] * 100,
    })
    kept, (val,), dropped = prune_features(df, [df.copy()])
    assert "good" in kept.columns
    # half-missing survives: sparse aggregates are weak, not worthless
    assert "half_missing" in kept.columns
    assert "constant" in dropped and "all_missing" in dropped
    assert list(val.columns) == list(kept.columns)


def test_prune_validates_thresholds():
    from tabicl.scaling import prune_features

    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="max_missing"):
        prune_features(df, max_missing=1.5)
    with pytest.raises(ValueError, match="max_dominant"):
        prune_features(df, max_dominant=0.0)


def test_oversized_join_fails_fast_instead_of_exhausting_memory():
    """|child| x rows-sharing-a-key is invisible in the inputs; three runs died on it."""
    from tabicl.scaling import _relational

    entity = pd.DataFrame({"uid": [0] * 400})
    child = pd.DataFrame({"uid": [0] * 400, "v": np.arange(400.0)})
    original = _relational.MAX_JOIN_PAIRS
    _relational.MAX_JOIN_PAIRS = 1000  # 400 x 400 = 160,000 pairs
    try:
        with pytest.raises(MemoryError, match="asof_statistics"):
            flatten_relational(entity, "uid", [Table(child, "uid", "ev")])
    finally:
        _relational.MAX_JOIN_PAIRS = original


def test_estimate_matches_the_actual_join_size():
    from tabicl.scaling._relational import _estimate_pairs

    entity = pd.DataFrame({"uid": [0, 0, 1, 2]})
    child = pd.DataFrame({"uid": [0, 0, 0, 1], "v": [1.0, 2.0, 3.0, 4.0]})
    anchor = pd.DataFrame({"__key": entity["uid"].values, "__row": np.arange(4)})
    # key 0: 3 child x 2 entity = 6; key 1: 1 x 1 = 1; key 2: none
    assert _estimate_pairs(Table(child, "uid", "ev"), anchor) == 7
    actual = len(child.merge(anchor, left_on="uid", right_on="__key"))
    assert actual == 7


def test_asof_handles_pandas_nullable_dtypes():
    """Int64/Float64 masked arrays refuse a plain float64 conversion when null."""
    from tabicl.scaling import asof_statistics

    ent = pd.DataFrame({"uid": [0, 1], "cut": pd.to_datetime(["2026-02-01"] * 2)})
    ch = pd.DataFrame({
        "uid": [0, 0, 1],
        "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "v": pd.array([1, None, 3], dtype="Int64"),
        "w": pd.array([None, None, None], dtype="Float64"),
    })
    out = asof_statistics(Table(ch, "uid", "ev", time_column="ts"),
                          ent["uid"].to_numpy(), ent["cut"].to_numpy())
    assert out["ev__count"].tolist() == [2.0, 1.0]
    # One null used to poison every prefix after it via cumsum, silently returning
    # all-NaN for the column. Nulls now contribute zero and leave the denominator.
    assert out["ev__v__count"].iloc[0] == 1.0
    assert out["ev__v__mean"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(out["ev__w__mean"].iloc[0])      # all-null stays unknown


def test_max_columns_keeps_the_best_populated_sources():
    """The budget must cut by coverage, not by column order.

    rel-event needed 35.7 GB because every column of a 2.5M-row table became several
    statistics. A cap that just took the first N would as happily keep an all-null
    column and drop a full one.
    """
    entity = pd.DataFrame({"uid": [0, 1], "cut": pd.to_datetime(["2026-02-01"] * 2)})
    child = pd.DataFrame(
        {
            "uid": [0, 0, 1],
            "ts": pd.to_datetime(["2026-01-01"] * 3),
            "sparse": [1.0, np.nan, np.nan],   # first in order, worst coverage
            "full": [1.0, 2.0, 3.0],
            "half": [1.0, np.nan, 3.0],
        }
    )
    table = Table(child, "uid", "ev", time_column="ts", max_columns=2)
    out = flatten_relational(entity, "uid", [table], cutoff_column="cut")

    sources = {c.split("__")[1] for c in out.columns if c.startswith("ev__") and c != "ev__count"}
    assert sources == {"full", "half"}, sources

    # Values must be untouched by the presence of a budget.
    assert out.loc[out.index[0], "ev__full__mean"] == pytest.approx(1.5)


def test_max_columns_applies_to_the_asof_path_too():
    """Both aggregation paths share one budget decision, or the arms stop matching."""
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0, 0, 1],
            "ts": pd.to_datetime(["2026-01-01"] * 3),
            "sparse": [np.nan, np.nan, 5.0],
            "full": [1.0, 2.0, 3.0],
        }
    )
    keys = np.array([0, 1])
    cutoffs = np.array(pd.to_datetime(["2026-02-01"] * 2))

    capped = asof_statistics(Table(child, "uid", "ev", time_column="ts", max_columns=1), keys, cutoffs)
    uncapped = asof_statistics(Table(child, "uid", "ev", time_column="ts"), keys, cutoffs)

    assert [c for c in capped.columns if c.endswith("__mean")] == ["ev__full__mean"]
    assert any("sparse" in c for c in uncapped.columns)
    # The kept column's statistics are identical either way.
    pd.testing.assert_series_equal(capped["ev__full__mean"], uncapped["ev__full__mean"])


def test_max_columns_above_the_column_count_is_a_no_op():
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {"uid": [0, 1], "ts": pd.to_datetime(["2026-01-01"] * 2), "a": [1.0, 2.0]}
    )
    keys, cutoffs = np.array([0, 1]), np.array(pd.to_datetime(["2026-02-01"] * 2))
    wide = asof_statistics(Table(child, "uid", "ev", time_column="ts", max_columns=99), keys, cutoffs)
    plain = asof_statistics(Table(child, "uid", "ev", time_column="ts"), keys, cutoffs)
    pd.testing.assert_frame_equal(wide, plain)


def test_category_histogram_keeps_the_distribution_not_the_centroid():
    """Two entities with identical child counts but different category mixes.

    The point of the block is that these must not look alike. A mean over any encoding
    of the category would put both near the global centre; proportions separate them.
    """
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0] * 5 + [1] * 4,
            "ts": pd.to_datetime(["2026-01-01"] * 9),
            "kind": ["elec", "elec", "elec", "fit", "fit", "book", "book", "kitch", "kitch"],
        }
    )
    keys = np.array([0, 1])
    cutoffs = np.array(pd.to_datetime(["2026-02-01"] * 2))
    out = asof_statistics(Table(child, "uid", "ev", time_column="ts", top_k_categories=4), keys, cutoffs)

    from tabicl.scaling._relational import _category_codebook

    codebook = _category_codebook(child, "kind", 4)
    prop = {v: out[f"ev__kind__cat{j}"].to_numpy() for j, v in enumerate(codebook)}

    assert prop["elec"][0] == pytest.approx(0.6)
    assert prop["fit"][0] == pytest.approx(0.4)
    assert prop["book"][1] == pytest.approx(0.5)
    assert prop["kitch"][1] == pytest.approx(0.5)
    # Nothing outside the codebook, so proportions account for every row.
    cats = [c for c in out.columns if "__cat" in c]
    assert out[cats].sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0])


def test_category_histogram_respects_the_cutoff():
    """A category appearing only after the cutoff must not reach the features."""
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0, 0],
            "ts": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "kind": ["before", "after"],
        }
    )
    out = asof_statistics(
        Table(child, "uid", "ev", time_column="ts", top_k_categories=2),
        np.array([0]),
        np.array(pd.to_datetime(["2026-02-01"])),
    )
    from tabicl.scaling._relational import _category_codebook

    codebook = _category_codebook(child, "kind", 2)
    prop = {v: out[f"ev__kind__cat{j}"].to_numpy()[0] for j, v in enumerate(codebook)}
    assert prop["before"] == pytest.approx(1.0)
    assert prop["after"] == pytest.approx(0.0)


def test_category_histogram_works_over_windows():
    """The property mode cannot have.

    ``mode`` has no linear range algorithm, so the join path can only offer it over all
    history. Counts are invertible, so this block is a prefix difference and a window
    costs one extra lookup.
    """
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0] * 4,
            # two old "cold" rows, two recent "hot" rows
            "ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2026-01-20", "2026-01-25"]),
            "kind": ["cold", "cold", "hot", "hot"],
        }
    )
    table = Table(
        child, "uid", "ev", time_column="ts",
        windows=[pd.Timedelta(days=30)], top_k_categories=2,
    )
    out = asof_statistics(table, np.array([0]), np.array(pd.to_datetime(["2026-02-01"])))

    from tabicl.scaling._relational import _category_codebook

    codebook = _category_codebook(child, "kind", 2)
    idx = {v: j for j, v in enumerate(codebook)}
    stems = {c.rsplit("__kind__", 1)[0] for c in out.columns if "__kind__cat" in c}
    stem = next(s for s in stems if s != "ev")   # the window block, not all-history
    assert stem == "ev_30d", stem

    # All history: half cold, half hot. Last 30 days: entirely hot.
    assert out[f"ev__kind__cat{idx['cold']}"].to_numpy()[0] == pytest.approx(0.5)
    assert out[f"{stem}__kind__cat{idx['hot']}"].to_numpy()[0] == pytest.approx(1.0)
    assert out[f"{stem}__kind__cat{idx['cold']}"].to_numpy()[0] == pytest.approx(0.0)


def test_category_histogram_buckets_the_tail_and_exposes_missingness():
    """Values outside the codebook become tail mass; nulls show up as a shortfall."""
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0] * 4,
            "ts": pd.to_datetime(["2026-01-01"] * 4),
            "kind": ["top", "top", "rare", None],
        }
    )
    out = asof_statistics(
        Table(child, "uid", "ev", time_column="ts", top_k_categories=1),
        np.array([0]),
        np.array(pd.to_datetime(["2026-02-01"])),
    )
    assert out["ev__kind__cat0"].to_numpy()[0] == pytest.approx(0.5)      # 2 of 4
    assert out["ev__kind__catother"].to_numpy()[0] == pytest.approx(0.25)  # "rare"
    # The missing row is in the denominator but no bucket, so the shortfall is the
    # missing rate rather than silently inflating the categories that are present.
    cats = [c for c in out.columns if "__cat" in c]
    assert out[cats].sum(axis=1).to_numpy()[0] == pytest.approx(0.75)


def test_category_histogram_is_off_by_default():
    """Every category costs a column, so it must be opt-in."""
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {"uid": [0, 1], "ts": pd.to_datetime(["2026-01-01"] * 2), "kind": ["a", "b"]}
    )
    out = asof_statistics(
        Table(child, "uid", "ev", time_column="ts"),
        np.array([0, 1]),
        np.array(pd.to_datetime(["2026-02-01"] * 2)),
    )
    assert not [c for c in out.columns if "__cat" in c]


def test_category_codebook_is_deterministic_under_ties():
    """Column j must denote the same category on every run, or blocks aren't comparable."""
    from tabicl.scaling._relational import _category_codebook

    df = pd.DataFrame({"v": ["b", "b", "a", "a", "c", "c"]})   # a three-way tie
    shuffled = df.iloc[::-1].reset_index(drop=True)
    assert _category_codebook(df, "v", 3) == _category_codebook(shuffled, "v", 3)


def test_prefix_mode_matches_the_join_path_on_random_data():
    """The vectorised running argmax must agree with the join path's mode.

    The scan version replaces a sequential loop with three segmented passes, on the
    argument that a leader only changes when some count strictly exceeds every count
    before it. Randomised keys, values, ties and nulls are what makes that argument
    testable rather than merely plausible.
    """
    from tabicl.scaling import asof_statistics

    rng = np.random.default_rng(0)
    n = 4000
    child = pd.DataFrame(
        {
            "uid": rng.integers(0, 60, n),
            "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
            "kind": rng.choice(["a", "b", "c", "d", None], size=n),
        }
    )
    entity = pd.DataFrame({"uid": np.arange(60), "cut": pd.to_datetime("2027-01-01")})

    scanned = asof_statistics(
        Table(child, "uid", "ev", time_column="ts", include_mode=True),
        entity["uid"].to_numpy(),
        entity["cut"].to_numpy(),
    )
    joined = flatten_relational(
        entity, "uid", [Table(child, "uid", "ev", time_column="ts")], cutoff_column="cut"
    )

    codes, uniques = pd.factorize(child["kind"], use_na_sentinel=True)
    got = [uniques[int(v)] if pd.notna(v) else None for v in scanned["ev__kind__mode"]]
    want = [v if pd.notna(v) else None for v in joined["ev__kind__mode"]]

    # Tied groups are excluded, and the exclusion is the point: with a tie there is no
    # single mode, and the two paths resolve it by different deterministic rules -- the
    # scan takes the first value to reach the leading count in time order, the join path
    # takes the smallest value. Asserting agreement there would be asserting an
    # arbitrary choice. Everywhere the mode is actually well defined they must agree.
    unambiguous = []
    for uid in entity["uid"]:
        counts = child.loc[child["uid"] == uid, "kind"].value_counts()
        unambiguous.append(len(counts) > 0 and (counts == counts.max()).sum() == 1)

    assert any(unambiguous), "test data degenerated to all-ties"
    for i, ok in enumerate(unambiguous):
        if ok:
            assert got[i] == want[i], f"uid={entity['uid'][i]}: scan={got[i]} join={want[i]}"


def test_prefix_mode_ignores_nulls_and_empty_prefixes():
    from tabicl.scaling import asof_statistics

    child = pd.DataFrame(
        {
            "uid": [0, 0, 0, 1],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-03-01"]),
            "kind": [None, "z", None, "q"],
        }
    )
    out = asof_statistics(
        Table(child, "uid", "ev", time_column="ts", include_mode=True),
        np.array([0, 1]),
        np.array(pd.to_datetime(["2026-02-01"] * 2)),
    )
    codes, uniques = pd.factorize(child["kind"], use_na_sentinel=True)
    # uid 0: nulls cannot lead, so the only non-null value wins.
    assert uniques[int(out["ev__kind__mode"].to_numpy()[0])] == "z"
    # uid 1: its only row is after the cutoff, so the prefix is empty.
    assert np.isnan(out["ev__kind__mode"].to_numpy()[1])


def test_sweep_picks_the_cheapest_within_tolerance_not_the_argmax():
    """The whole reason to calibrate a cost knob is to spend less.

    Selecting the argmax gives the saving straight back and chases validation noise, so
    the rule is the cheapest candidate that is good enough.
    """
    from tabicl.scaling import sweep_configurations

    scores = {1000: 0.800, 5000: 0.803, 20000: 0.804}
    result = sweep_configurations(list(scores), lambda c: scores[c], tolerance=0.005)

    assert result.chosen == 1000        # within 0.005 of the best, and cheapest
    assert result.best == 20000
    assert result.chosen_score == pytest.approx(0.800)
    assert result.best_score == pytest.approx(0.804)


def test_sweep_with_zero_tolerance_is_the_argmax():
    from tabicl.scaling import sweep_configurations

    scores = {1000: 0.800, 5000: 0.803, 20000: 0.804}
    result = sweep_configurations(list(scores), lambda c: scores[c], tolerance=0.0)
    assert result.chosen == 20000 == result.best


def test_sweep_respects_a_genuinely_steep_curve():
    """A knob that matters must not be shrunk away.

    max_columns=2 cost rel-f1 19.5 points. A calibration that still chose the cheap end
    on a curve like that would be worse than no calibration at all.
    """
    from tabicl.scaling import sweep_configurations

    scores = {1000: 0.60, 5000: 0.72, 20000: 0.80}
    result = sweep_configurations(list(scores), lambda c: scores[c], tolerance=0.005)
    assert result.chosen == 20000


def test_calibrate_context_size_shrinks_a_flat_curve():
    """End to end, with a scorer that ignores context size: take the smallest."""
    from tabicl.scaling import calibrate_context_size

    rng = np.random.default_rng(0)
    X_tr, y_tr = rng.normal(size=(400, 3)), rng.integers(0, 2, 400)
    X_va, y_va = rng.normal(size=(60, 3)), rng.integers(0, 2, 60)

    seen = []

    def fit_score(Xc, yc, Xv, yv):
        seen.append(len(Xc))
        return 0.75

    result = calibrate_context_size(
        X_tr, y_tr, X_va, y_va, fit_score, candidates=(50, 100, None), tolerance=0.005
    )
    assert result.chosen == 50
    assert seen[0] == 50 and seen[-1] == 400          # None means the full pool
    assert len(result.curve) == 3


def test_calibrate_context_size_collapses_duplicate_sizes():
    """Candidates at or above the pool size are the same experiment; run it once."""
    from tabicl.scaling import calibrate_context_size

    rng = np.random.default_rng(0)
    X_tr, y_tr = rng.normal(size=(80, 2)), rng.integers(0, 2, 80)
    X_va, y_va = rng.normal(size=(20, 2)), rng.integers(0, 2, 20)

    calls = []
    result = calibrate_context_size(
        X_tr, y_tr, X_va, y_va,
        lambda Xc, yc, Xv, yv: (calls.append(len(Xc)), 0.5)[1],
        candidates=(1000, 5000, None),
    )
    assert len(result.curve) == 1
    assert calls == [80]


def test_calibrate_context_size_stratifies_by_default():
    """At small budgets on a skewed target, an unstratified draw measures the draw.

    rel-avito's positive rate is 0.905; this asserts the minority class survives.
    """
    from tabicl.scaling import calibrate_context_size

    rng = np.random.default_rng(0)
    n = 2000
    y_tr = (rng.random(n) < 0.905).astype(int)
    X_tr = rng.normal(size=(n, 2))
    X_va, y_va = rng.normal(size=(50, 2)), rng.integers(0, 2, 50)

    rates = []
    calibrate_context_size(
        X_tr, y_tr, X_va, y_va,
        lambda Xc, yc, Xv, yv: (rates.append(float(yc.mean())), 0.5)[1],
        candidates=(100,),
    )
    assert rates[0] == pytest.approx(y_tr.mean(), abs=0.02)


def test_sweep_declines_infeasible_candidates_without_running_them():
    """Catching MemoryError is not a memory guard.

    A configuration larger than RAM does not reliably raise on a paging OS -- it
    thrashes, and takes the machine with it. A rel-avito candidate reached 24.6 GB on a
    5,000-row context because the as-of scan sizes its prefix arrays by the child table,
    not the context. So the expensive candidate must be declined *before* it runs.
    """
    from tabicl.scaling import sweep_configurations

    ran = []

    def score(candidate):
        ran.append(candidate)
        return {"small": 0.70, "huge": 0.99}[candidate]

    result = sweep_configurations(
        ["small", "huge"], score, feasible=lambda c: c != "huge"
    )
    assert ran == ["small"], "the infeasible candidate must not be evaluated"
    assert result.chosen == "small"
    assert dict(result.curve)["huge"] == float("-inf")


def test_sweep_raises_when_nothing_is_runnable():
    """Silently returning an unrunnable setting would be worse than failing."""
    from tabicl.scaling import sweep_configurations

    with pytest.raises(RuntimeError, match="no candidate was runnable"):
        sweep_configurations(["a", "b"], lambda c: 1.0, feasible=lambda c: False)


def test_permutation_control_catches_a_feature_that_reads_its_own_label():
    """The canonical leak: a 0-hop term, a self-loop, or the target left in the matrix."""
    from tabicl.scaling import permutation_control

    y = np.array([0, 1] * 50)

    def leaky(labels):          # perfect score whatever the labels are
        return 1.0

    report = permutation_control(leaky, y, n_permutations=3)
    assert not report.passed
    assert "reading the row's own label" in report.reason


def test_permutation_control_passes_an_honest_feature():
    from tabicl.scaling import permutation_control

    y = np.array([0, 1] * 50)
    truth = y.copy()

    def honest(labels):
        # Scores well only when the labels it is handed are the real ones.
        return 0.95 if np.array_equal(labels, truth) else 0.5

    report = permutation_control(honest, y, n_permutations=3)
    assert report.passed
    assert report.observed == pytest.approx(0.95)


def test_temporal_control_catches_features_reaching_past_the_cutoff():
    """Withholding history cannot add information, so an improvement is a leak."""
    from tabicl.scaling import temporal_control

    report = temporal_control(lambda shift: 0.70 + 0.05 * shift, shifts=(0.0, 1.0, 2.0))
    assert not report.passed
    assert "reaching past the cutoff" in report.reason


def test_temporal_control_passes_when_earlier_cutoffs_degrade():
    from tabicl.scaling import temporal_control

    # 0.03 per DAY over a 30-day shift gave -0.10, which is not a possible ROC-AUC. The
    # raw-score comparison accepted it silently; comparing |score - chance| does not, since
    # |-0.10 - 0.5| = 0.6 is more information than |0.80 - 0.5| = 0.3. The rate is now per
    # day at a realistic scale, and temporal_control rejects impossible scores outright.
    report = temporal_control(lambda shift: 0.80 - 0.003 * shift, shifts=(0.0, 30.0))
    assert report.passed
    assert report.observed == pytest.approx(0.80)


def test_temporal_control_rejects_a_score_that_cannot_be_an_auc():
    from tabicl.scaling import temporal_control

    # A caller whose scoring function returns something outside [0, 1] has a bug, and
    # silently reading it as extreme information is the worst available response -- that is
    # how a test in this file asserted a pass on an AUC of -0.10 for months.
    with pytest.raises(ValueError, match="outside"):
        temporal_control(lambda shift: 0.80 - 0.03 * shift, shifts=(0.0, 30.0))


def test_temporal_control_requires_an_unshifted_baseline():
    from tabicl.scaling import temporal_control

    with pytest.raises(ValueError, match="must start at 0.0"):
        temporal_control(lambda s: 0.5, shifts=(30.0, 90.0))


def test_label_homophily_separates_signal_from_chance():
    """The gate: connected nodes sharing labels more than a random assignment would."""
    from tabicl.scaling import label_homophily

    labels = np.array([0, 0, 0, 1, 1, 1])
    homophilous = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])       # only within-class
    mixed = np.array([[0, 3], [1, 4], [2, 5]])                      # only across-class

    good = label_homophily(homophilous, labels)
    assert good["observed"] == pytest.approx(1.0)
    assert good["lift"] > 0.4

    bad = label_homophily(mixed, labels)
    assert bad["observed"] == pytest.approx(0.0)
    assert bad["lift"] < 0          # heterophily, the opposite signal


def test_graph_context_prefers_neighbours_of_many_queries():
    from tabicl.scaling import select_graph_context

    # node 0 neighbours both queries (10, 11); node 1 neighbours only one; 2 and 3 neither
    edges = np.array([[10, 0], [11, 0], [10, 1]])
    chosen = select_graph_context(edges, train_idx=np.array([0, 1, 2, 3]),
                                  query_idx=np.array([10, 11]), n_context=2)
    assert set(chosen.tolist()) == {0, 1}
    assert chosen[0] == 0, "the node reached by both queries must rank first"


def test_graph_context_never_puts_a_query_in_its_own_context():
    """A query in its own context is the leak this whole family is prone to."""
    from tabicl.scaling import select_graph_context

    edges = np.array([[0, 1], [1, 2]])
    chosen = select_graph_context(edges, train_idx=np.array([0, 1, 2]),
                                  query_idx=np.array([1]), n_context=2)
    assert 1 not in chosen.tolist()


def test_graph_context_returns_the_requested_size_even_when_isolated():
    """Equal context length is what makes the comparison against random meaningful."""
    from tabicl.scaling import select_graph_context

    edges = np.array([[0, 1]])
    chosen = select_graph_context(edges, train_idx=np.arange(10),
                                  query_idx=np.array([7]), n_context=4)
    assert len(chosen) == 4
    assert len(set(chosen.tolist())) == 4


def test_graph_context_two_hops_reaches_further_than_one():
    from tabicl.scaling import select_graph_context

    edges = np.array([[9, 0], [0, 1], [1, 2]])
    one = select_graph_context(edges, np.array([0, 1, 2]), np.array([9]), n_context=1, hops=1)
    two = select_graph_context(edges, np.array([0, 1, 2]), np.array([9]), n_context=2, hops=2)
    assert one.tolist() == [0]
    assert set(two.tolist()) == {0, 1}


def test_two_hop_cutoff_excludes_grandchildren_dated_after_the_entity_cutoff():
    """The existing two-hop test cannot catch this, and the leak it misses was live.

    `test_two_hop_cutoff_propagates` dates every grandchild identically to its parent, so
    excluding a grandchild is indistinguishable from excluding its parent. Here the parent
    order is comfortably before the cutoff and one of its two items is after it: only a
    deadline that actually reaches the grandchild level can tell the difference.
    """
    from tabicl.scaling import Table, flatten_relational

    entities = pd.DataFrame({"user": [1], "ts": [pd.Timestamp("2026-01-10")]})
    orders = pd.DataFrame({"oid": [10], "user": [1], "ots": [pd.Timestamp("2026-01-05")]})
    items = pd.DataFrame({
        "oid": [10, 10],
        "price": [2.0, 1000.0],
        "its": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-01")],  # second is after
    })

    item_tbl = Table(items, "oid", "item", time_column="its")
    order_tbl = Table(orders, "user", "ord", time_column="ots",
                      primary_key="oid", children=[item_tbl])
    out = flatten_relational(entities, "user", [order_tbl], cutoff_column="ts")

    count = next(c for c in out.columns if c.endswith("item__count"))
    assert out[count].iloc[0] == 1, "a grandchild dated after the cutoff was counted"
    mean = next(c for c in out.columns if c.endswith("price__mean"))
    assert out[mean].iloc[0] == pytest.approx(2.0), "the post-cutoff price leaked into mean"


def test_column_budget_does_not_spend_a_slot_on_a_constant_column():
    """Coverage ranking actively prefers constants: a constant is 100% populated.

    Found on rel-trial, where `designs.subject_masked` is 't' in every row of the table.
    With `numeric_booleans` on it became a float column, won a slot at `max_columns=2` on
    perfect coverage, and produced a mean of exactly 1.000 for every entity -- which is
    what the boolean rates were about to be measured on.

    Fewer varying columns than the budget is a reason to emit fewer, not to pad with
    constants: all four masked flags in that table are constant, so a rule requiring
    `max_columns` survivors put one back every time.

    Target-free, so it cannot leak -- this reads the feature column and never y. And it
    changes nothing on the default path: no task has a constant numeric child column.
    """
    from tabicl.scaling import Table, asof_statistics

    entities = pd.DataFrame({"id": [1], "t": [pd.Timestamp("2020-06-01")]})
    kid = pd.DataFrame({
        "id": [1, 1, 1],
        "kt": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "constant": [7.0, 7.0, 7.0],           # fully populated, and worthless
        "informative": [1.0, 2.0, np.nan],     # worse coverage, and not worthless
    })

    out = asof_statistics(
        Table(kid, "id", "k", time_column="kt", max_columns=1),
        entities["id"].to_numpy(), entities["t"].to_numpy())

    assert not [c for c in out.columns if "constant" in c], "a constant took the slot"
    assert [c for c in out.columns if "informative" in c]

    # All-constant candidates: emit nothing rather than pad, since nothing can help.
    both_constant = pd.DataFrame({
        "id": [1, 1, 1], "kt": kid["kt"], "a": [7.0, 7.0, 7.0], "b": [8.0, 8.0, 8.0],
    })
    degenerate = asof_statistics(
        Table(both_constant, "id", "k", time_column="kt", max_columns=1),
        entities["id"].to_numpy(), entities["t"].to_numpy())
    assert "k__count" in degenerate.columns          # the row count still means something


def test_calendar_features_are_cyclical_and_keep_the_trend_separate():
    """Sunday and Monday are adjacent, and the monotone column is opt-in.

    Both runners drop every datetime column when assembling the entity block, the cutoff
    included, so nothing downstream can tell a Monday from a Saturday -- on tasks whose
    horizons are four and seven days.

    `trend` is separate on purpose: days-since-origin is monotone, so every test row lies
    beyond the training range on it. Bundling it with the cyclical features would let the
    pair win or lose for reasons that cannot be told apart.
    """
    from tabicl.scaling._calendar import calendar_features

    # 2020-01-05 is a Sunday, 2020-01-06 a Monday.
    stamps = pd.to_datetime(["2020-01-05", "2020-01-06", "2020-06-15"]).to_numpy()

    plain = calendar_features(stamps)
    assert "cal__trend" not in plain.columns
    assert plain["cal__dayofweek"].tolist() == [6.0, 0.0, 0.0]
    assert plain["cal__is_weekend"].tolist() == [1.0, 0.0, 0.0]

    # The whole point of the sine/cosine pair: consecutive days are close in that space,
    # even though 6 and 0 are far apart as integers.
    def gap(i, j):
        return float(np.hypot(plain["cal__dow_sin"].iloc[i] - plain["cal__dow_sin"].iloc[j],
                              plain["cal__dow_cos"].iloc[i] - plain["cal__dow_cos"].iloc[j]))

    assert gap(0, 1) < 1.0                      # Sunday to Monday: one step round the circle
    assert gap(0, 1) < abs(plain["cal__dayofweek"].iloc[0] - plain["cal__dayofweek"].iloc[1])

    # Trend is measured from the given origin, so train and test share a scale.
    origin = pd.Timestamp("2020-01-01")
    trended = calendar_features(stamps, trend=True, origin=origin)
    assert trended["cal__trend"].tolist() == [4.0, 5.0, 166.0]


def test_recency_age_and_span_are_available_and_off_by_default():
    """Nothing the scan emits says *when*, because the timestamp is the excluded column.

    Every other statistic answers how much or what kind. On rel-avito, which asks whether
    a user visits in the next four days, a count over a 7-day window cannot separate a
    user who searched once yesterday from one who searched once six days ago.

    Over a window, `age` pins to the window edge for anyone active throughout, which makes
    it a "was already here" indicator rather than a second copy of recency -- checked here,
    since that is the part that is easy to get wrong.
    """
    from tabicl.scaling import Table, asof_statistics

    kid = pd.DataFrame({
        "id": [1, 1, 1, 1, 2],
        "kt": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-25", "2020-05-30",
                              "2020-01-01"]),
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    keys = np.array([1, 2, 3])                       # entity 3 has no history at all
    cutoffs = pd.to_datetime(["2020-06-01"] * 3).to_numpy()

    default = asof_statistics(Table(kid, "id", "k", time_column="kt"), keys, cutoffs)
    assert not [c for c in default.columns if c.endswith("__recency")]

    out = asof_statistics(
        Table(kid, "id", "k", time_column="kt", time_deltas=True,
              windows=[pd.Timedelta(days=30)]), keys, cutoffs)

    assert out["k__recency"].iloc[0] == pytest.approx(2.0)     # last row 2020-05-30
    assert out["k__age"].iloc[0] == pytest.approx(152.0)       # first row 2020-01-01
    assert out["k__span"].iloc[0] == pytest.approx(150.0)
    # Within 30 days only 05-25 and 05-30 survive, so age is 7 and not 152.
    assert out["k_30d__age"].iloc[0] == pytest.approx(7.0)
    assert out["k_30d__span"].iloc[0] == pytest.approx(5.0)

    # A single event has a recency but no span.
    assert out["k__recency"].iloc[1] == pytest.approx(152.0)
    assert out["k__span"].iloc[1] == pytest.approx(0.0)

    # No history means no recency. Zero would assert the opposite of what is true.
    assert np.isnan(out["k__recency"].iloc[2])
    assert out["k__count"].iloc[2] == pytest.approx(0.0)
    assert np.isnan(out["k_30d__recency"].iloc[1])             # active, but not lately

    # An empty child table: every range is empty, and the gather must not index it.
    empty = asof_statistics(
        Table(kid.iloc[:0], "id", "k", time_column="kt", time_deltas=True), keys, cutoffs)
    assert empty["k__recency"].isna().all()


def test_leak_guard_fires_on_a_passed_through_target_but_not_on_a_strong_feature():
    """A leak that scores 100.00 in *both* arms of a paired comparison reads as +0.00.

    That is what `eval_depth2` was reporting: `flatten_relational` passes through every
    entity column except the key and the cutoff, so the target rode along as a feature.
    Both depths scored AUC 100.00 and their difference was a clean +0.00 with sd 0.00 --
    a comparison of two leaks, indistinguishable in the output from a careful null.

    The guard tests the symptom rather than the cause, so it does not need to know which
    mistake produced it: a renamed target, a duplicated column, or an aggregate that
    reconstructs it all separate the labels perfectly and all get caught.
    """
    from tabicl.scaling._guards import assert_no_perfect_feature

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    noise = rng.normal(size=(400, 3))

    # Must not fire on an honest strong feature -- a guard that cries wolf gets disabled.
    strong = y + rng.normal(scale=0.6, size=400)
    assert_no_perfect_feature(np.column_stack([noise, strong]), y, ["a", "b", "c", "s"])

    with pytest.raises(SystemExit, match="leak"):
        assert_no_perfect_feature(np.column_stack([noise, y]), y, ["a", "b", "c", "target"])

    # Perfectly anti-correlated is exactly as much of a leak.
    with pytest.raises(SystemExit, match="leak"):
        assert_no_perfect_feature(np.column_stack([noise, -y]), y, ["a", "b", "c", "neg"])


def test_max_columns_bounds_numeric_columns_only_unless_told_otherwise():
    """One table, budgeted three different ways -- which is worth pinning down.

    `max_columns` has never applied to the `nunique` block, while the category histogram
    and `include_mode` both apply it. So the budget bounds numeric columns only, and a wide
    categorical table emits a distinct-count per column however narrow the budget is. That
    is the failure the budget exists to prevent; rel-event reached 1,670 features that way.

    Asserted in both directions, because the default is the *unbudgeted* one and a silent
    change there moves every standing number.
    """
    from tabicl.scaling import Table, asof_statistics

    entities = pd.DataFrame({"id": [1], "t": [pd.Timestamp("2020-06-01")]})
    kid = pd.DataFrame({
        "id": [1, 1],
        "kt": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "c1": ["a", "b"], "c2": ["c", "d"], "c3": ["e", "f"], "c4": ["g", "h"],
    })

    def widths(budget_categoricals):
        table = Table(kid, "id", "k", time_column="kt", max_columns=2,
                      budget_categoricals=budget_categoricals)
        out = asof_statistics(table, entities["id"].to_numpy(), entities["t"].to_numpy())
        return [c for c in out.columns if c.endswith("__nunique")]

    # Default: the budget of 2 does not reach this block, so all four columns appear.
    assert len(widths(False)) == 4
    assert len(widths(True)) == 2


def test_boolean_columns_can_carry_their_rate_instead_of_a_nunique():
    """The mean of a boolean is its rate, and that is the statistic a boolean history has.

    Booleans are categorical by default, which gives them exactly one feature: a `nunique`
    that is 1 or 2. With `include_mode` and the histogram both off -- which is how every
    standing number was measured -- that is the entire contribution of a boolean child
    column. `numeric_booleans` routes them through the numeric path instead.

    The nullable `boolean` dtype is covered deliberately: `astype("float64")` raises on it
    outright.

    A flag is recognised by its values, not its dtype, and that distinction is the whole
    feature here: RelBench spells every boolean it has as `'t'`/`'f'` in an object column
    -- `eligibilities.adult`, `designs.subject_masked`, `studies.is_fda_regulated_drug`
    and eight more on rel-trial alone. Keying off `is_bool_dtype` made this a no-op on the
    entire benchmark, and only the empty-block refusal surfaced that rather than a +0.00.
    """
    from tabicl.scaling import Table, asof_statistics

    entities = pd.DataFrame({"id": [1], "t": [pd.Timestamp("2020-06-01")]})
    kid = pd.DataFrame({
        "id": [1, 1, 1],
        "kt": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "flag": np.array([True, False, True]),
        "tf": ["t", "f", "t"],                       # how RelBench actually spells it
        "yesno": ["Yes", "No", "Yes"],
        "nullable": pd.array([True, None, True], dtype="boolean"),
        "genuine_category": ["a", "b", "c"],         # three values: must stay categorical
    })

    def run(numeric_booleans):
        table = Table(kid, "id", "k", time_column="kt", numeric_booleans=numeric_booleans)
        return asof_statistics(table, entities["id"].to_numpy(), entities["t"].to_numpy())

    default = run(False)
    assert [c for c in default.columns if "flag" in c] == ["k__flag__nunique"]
    assert "k__tf__nunique" in default.columns

    numeric = run(True)
    for column in ("flag", "tf", "yesno"):
        assert f"k__{column}__nunique" not in numeric.columns
        assert numeric[f"k__{column}__mean"].iloc[0] == pytest.approx(2 / 3)
    # Nulls leave the denominator rather than counting as False, so this is 2 of 2.
    assert numeric["k__nullable__mean"].iloc[0] == pytest.approx(1.0)
    assert numeric["k__nullable__count"].iloc[0] == pytest.approx(2.0)
    # A three-valued column is not a flag, whatever the flag setting says.
    assert "k__genuine_category__nunique" in numeric.columns


def test_column_budget_does_not_delete_the_grandchild_block():
    """`max_columns` caps a table's own source columns, never its nested statistics.

    Ranking both kinds together made depth-2 unmeasurable rather than merely narrow.
    Coverage prefers dense raw columns and a grandchild block is sparse by construction,
    so at `max_columns=2` the nested columns lost every slot and depth-2 emitted output
    byte-identical to depth-1. The measurement then read +0.00 with sd 0.00 over five
    seeds, which is indistinguishable from a genuine null and was in fact an empty block.

    A silently-empty feature block is worse than a crash: it reports a number.
    """
    from tabicl.scaling import Table, flatten_relational

    entities = pd.DataFrame({"user": [1, 2], "ts": [pd.Timestamp("2026-06-01")] * 2})
    orders = pd.DataFrame({
        "oid": [10, 11], "user": [1, 2], "ots": [pd.Timestamp("2026-01-01")] * 2,
        # Three dense raw columns, so a budget of 2 is genuinely oversubscribed.
        "a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0],
    })
    items = pd.DataFrame({
        "oid": [10, 10, 11], "price": [2.0, 4.0, 6.0],
        "its": [pd.Timestamp("2026-02-01")] * 3,
    })

    def build(depth, max_columns):
        kid = Table(items, "oid", "item", time_column="its", max_columns=2)
        spec = Table(orders, "user", "ord", time_column="ots", max_columns=max_columns,
                     primary_key="oid" if depth == 2 else None,
                     children=[kid] if depth == 2 else None)
        return flatten_relational(entities, "user", [spec], cutoff_column="ts")

    budgeted = build(2, max_columns=2)
    nested = [c for c in budgeted.columns if "item" in c]
    assert nested, "the column budget deleted the entire grandchild block"

    # The budget still binds on the child's own columns -- exempting nested statistics
    # must not quietly disable it.
    own = [c for c in budgeted.columns if c.startswith("ord__") and "item" not in c]
    assert not any(c.startswith("ord__c__") for c in own), "max_columns stopped applying"

    # And the nested block is whatever it would have been unbudgeted: a deeper level
    # carries its own cap, so exempting it here is bounded, not unbounded.
    assert nested == [c for c in build(2, max_columns=None).columns if "item" in c]

    # The failure this guards: depth-2 indistinguishable from depth-1.
    assert list(budgeted.columns) != list(build(1, max_columns=2).columns)


def test_key_history_subtracts_the_rows_own_outcome_rather_than_dropping_it():
    """Self-exclusion must remove one observation, not a key's whole accumulated history."""
    from tabicl.scaling import key_target_history

    base = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    # Three studies share sponsor "s". A and B succeeded, C is the query and also succeeded.
    links = pd.DataFrame({"study": ["A", "B", "C"], "sponsor": ["s", "s", "s"]})
    out = key_target_history(
        links,
        label_entities=np.array(["A", "B", "C"]),
        label_values=np.array([1.0, 1.0, 1.0]),
        label_times=np.array([base, base + day, base + 2 * day]),
        query_entities=np.array(["C"]),
        query_times=np.array([base + 10 * day]),
    )
    # C's own success is inside the window and must be subtracted, leaving A and B.
    assert out["hist__n_prior"].iloc[0] == 2, "own outcome not subtracted, or history dropped"
    assert out["hist__positive_rate"].iloc[0] == 1.0


def test_key_history_respects_the_resolution_horizon():
    """rel-trial's outcomes take 365 days to resolve; reading them earlier is the future."""
    from tabicl.scaling import key_target_history

    base = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    links = pd.DataFrame({"study": ["A", "B"], "sponsor": ["s", "s"]})
    kw = dict(label_entities=np.array(["A"]), label_values=np.array([1.0]),
              label_times=np.array([base]), query_entities=np.array(["B"]))

    early = key_target_history(links, query_times=np.array([base + 100 * day]),
                               label_horizon=365 * day, **kw)
    assert early["hist__n_prior"].iloc[0] == 0
    assert np.isnan(early["hist__positive_rate"].iloc[0])

    late = key_target_history(links, query_times=np.array([base + 400 * day]),
                              label_horizon=365 * day, **kw)
    assert late["hist__n_prior"].iloc[0] == 1
    assert late["hist__positive_rate"].iloc[0] == 1.0

    # Without the horizon the same early query reads an outcome a year from resolving.
    naive = key_target_history(links, query_times=np.array([base + 100 * day]), **kw)
    assert naive["hist__positive_rate"].iloc[0] == 1.0


def test_key_history_structural_degree_ignores_labels_entirely():
    """`n_linked` must be label-free, or it cannot serve as the leak-free comparison.

    Two studies share a sponsor; only one has an outcome, and it is outside the window.
    The label-derived counts are therefore 0 while the structural count is not.
    """
    from tabicl.scaling import key_target_history

    base = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    links = pd.DataFrame({"study": ["A", "B"], "sponsor": ["s", "s"]})
    kw = dict(links=links, label_entities=np.array(["A"]),
              label_times=np.array([base]), query_entities=np.array(["B"]),
              query_times=np.array([base + 10 * day]), label_horizon=365 * day)

    out = key_target_history(label_values=np.array([1.0]), **kw)
    assert out["hist__n_prior"].iloc[0] == 0, "the only outcome is still unresolved"
    assert out["hist__n_linked"].iloc[0] == 1, "but the sponsor link exists regardless"

    # Flipping the label must not move the structural count by even one.
    flipped = key_target_history(label_values=np.array([0.0]), **kw)
    assert flipped["hist__n_linked"].iloc[0] == out["hist__n_linked"].iloc[0]


def test_key_history_gives_nan_not_zero_for_no_track_record():
    from tabicl.scaling import key_target_history

    base = pd.Timestamp("2020-01-01")
    links = pd.DataFrame({"study": ["A", "Z"], "sponsor": ["s", "other"]})
    out = key_target_history(
        links,
        label_entities=np.array(["A"]), label_values=np.array([1.0]),
        label_times=np.array([base]),
        query_entities=np.array(["Z"]), query_times=np.array([base + pd.Timedelta(days=5)]),
    )
    assert out["hist__n_prior"].iloc[0] == 0
    assert np.isnan(out["hist__positive_rate"].iloc[0])


def test_neighbour_labels_never_include_the_rows_own_label():
    """The leak this family is prone to, pinned directly rather than inferred.

    Node 0 is positive and every other node is negative. If 0's own label reached its own
    feature, its positive rate would be non-zero.
    """
    from tabicl.scaling import neighbour_label_features

    edges = np.array([[0, 1], [0, 2], [1, 2], [0, 0]])   # including a self-loop
    nodes = np.array([0, 1, 2])
    values = np.array([1, 0, 0])
    out = neighbour_label_features(edges, nodes, values, query_nodes=nodes)

    assert out["nbr__positive_rate"].iloc[0] == 0.0, "node 0 saw its own positive label"
    # 1 and 2 are both neighbours of the positive node 0, and of each other.
    assert out["nbr__positive_rate"].iloc[1] == pytest.approx(0.5)
    assert out["nbr__labelled_degree"].tolist() == [2, 2, 2]


def test_isolated_rows_get_nan_not_a_zero_rate():
    """Absence of neighbours must not read as confident negativity."""
    from tabicl.scaling import neighbour_label_features

    edges = np.array([[0, 1]])
    out = neighbour_label_features(edges, np.array([0, 1]), np.array([1, 0]),
                                   query_nodes=np.array([0, 1, 9]))
    assert out["nbr__labelled_degree"].iloc[2] == 0
    assert np.isnan(out["nbr__positive_rate"].iloc[2])
    assert np.isnan(out["nbr__positive_count"].iloc[2])


def test_label_horizon_hides_outcomes_still_being_decided():
    """A neighbour's label is knowable only once its own window has closed."""
    from tabicl.scaling import neighbour_label_features

    edges = np.array([[0, 1]])
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2020-01-01")
    # Neighbour 1's label is stamped at day 0 and resolves over the following 10 days.
    kw = dict(label_nodes=np.array([1]), label_values=np.array([1]),
              label_times=np.array([base]), query_nodes=np.array([0]))

    early = neighbour_label_features(edges, query_times=np.array([base + 5 * day]),
                                     label_horizon=10 * day, **kw)
    assert early["nbr__labelled_degree"].iloc[0] == 0, "read a label still resolving"
    assert np.isnan(early["nbr__positive_rate"].iloc[0])

    late = neighbour_label_features(edges, query_times=np.array([base + 20 * day]),
                                    label_horizon=10 * day, **kw)
    assert late["nbr__labelled_degree"].iloc[0] == 1
    assert late["nbr__positive_rate"].iloc[0] == 1.0

    # Without the horizon the same query reads the unresolved label -- the failure mode.
    naive = neighbour_label_features(edges, query_times=np.array([base + 5 * day]), **kw)
    assert naive["nbr__positive_rate"].iloc[0] == 1.0


def test_a_neighbour_counts_once_however_many_label_events_it_has():
    """Otherwise the feature measures activity rather than label."""
    from tabicl.scaling import neighbour_label_features

    edges = np.array([[0, 1], [0, 2]])
    base = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    # Node 1 appears three times, node 2 once. Node 1's latest usable label is 0.
    out = neighbour_label_features(
        edges,
        label_nodes=np.array([1, 1, 1, 2]),
        label_values=np.array([1, 1, 0, 1]),
        label_times=np.array([base, base + day, base + 2 * day, base]),
        query_nodes=np.array([0]),
        query_times=np.array([base + 10 * day]),
    )
    assert out["nbr__labelled_degree"].iloc[0] == 2, "two neighbours, not four events"
    assert out["nbr__positive_count"].iloc[0] == 1.0
    assert out["nbr__positive_rate"].iloc[0] == pytest.approx(0.5)


def test_permutation_control_catches_a_deliberately_leaky_feature():
    """The control must fire on a feature that reads the row's own label.

    A control that never fails proves nothing, so it is checked against a known-bad
    feature as well as the real one.
    """
    from tabicl.scaling import neighbour_label_features
    from tabicl.scaling._leakage import permutation_control
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 200
    nodes = np.arange(n)
    # A homophilous ring: neighbours share labels, so the honest feature has real signal.
    y = (np.arange(n) // 20) % 2
    edges = np.column_stack([nodes, (nodes + 1) % n])

    def honest(labels):
        out = neighbour_label_features(edges, nodes, labels, query_nodes=nodes)
        score = out["nbr__positive_rate"].fillna(labels.mean()).to_numpy()
        return roc_auc_score(labels, score)

    def leaky(labels):
        return roc_auc_score(labels, labels.astype(float))    # reads the label itself

    assert permutation_control(honest, y, n_permutations=3).passed
    assert not permutation_control(leaky, y, n_permutations=3).passed


def test_unstratified_selection_can_collapse_the_class_balance():
    """The defect this guards against, stated as a test so it stays visible.

    Ranking by reach is ranking by popularity, and on a real social graph popularity
    tracks the label: on rel-event this returned a context at a 0.02-0.05 positive rate
    against a 0.163 base rate. Here every hub is negative, so an unstratified selection
    takes only negatives while the pool is half positive.
    """
    from tabicl.scaling import select_graph_context

    # Nodes 0-3 are negative hubs every query touches; 4-7 are positive leaves.
    queries = np.array([100, 101])
    edges = np.array([[q, h] for q in queries for h in range(4)]
                     + [[100, 4], [101, 5]])
    train_idx = np.arange(8)
    labels = np.zeros(102, dtype=np.int64)
    labels[4:8] = 1

    plain = select_graph_context(edges, train_idx, queries, n_context=4, hops=1)
    assert labels[plain].sum() == 0, "the hubs are all negative, so this is single-class"

    balanced = select_graph_context(edges, train_idx, queries, n_context=4, hops=1,
                                    labels=labels)
    assert len(balanced) == 4
    assert len(set(balanced.tolist())) == 4
    # The eligible pool is half positive, so the context must be too.
    assert labels[balanced].sum() == 2


def test_stratified_selection_still_prefers_graph_proximity_within_a_class():
    """Stratifying must not degrade into random selection -- it reorders within a class."""
    from tabicl.scaling import select_graph_context

    queries = np.array([100])
    # Positive 4 is a neighbour; positives 5, 6 are not. Negative 0 is a neighbour.
    edges = np.array([[100, 0], [100, 4]])
    train_idx = np.arange(7)
    labels = np.zeros(101, dtype=np.int64)
    labels[4:7] = 1

    chosen = select_graph_context(edges, train_idx, queries, n_context=2, hops=1,
                                  labels=labels, random_state=0)
    assert set(chosen.tolist()) == {0, 4}, "one per class, and the reachable one each time"


def test_every_scaling_module_parses_on_the_python_the_pods_run():
    """Syntax that is valid here and not on the pod costs a whole provisioning cycle.

    An f-string containing a literal newline parses on 3.12 (PEP 701) and is a syntax
    error on 3.10. One did, in `eval_graph_context`, and it was discovered only after a
    pod had been created, a payload uploaded, dependencies installed and a 385 MB dataset
    downloaded. Local Python is newer than the pod images, so nothing here would have
    caught it.

    `feature_version` makes the parser reject anything younger than the target, which is
    the whole check: this cannot run the code, only refuse to let it board.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tabicl" / "scaling"
    modules = sorted(root.glob("*.py"))
    assert modules, "no scaling modules found -- the path is wrong, not the code"

    failures = []
    for path in modules:
        try:
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 10))
        except SyntaxError as exc:
            failures.append(f"{path.name}:{exc.lineno}: {exc.msg}")
    assert not failures, "syntax too new for the pod images:\n" + "\n".join(failures)


def test_pod_images_satisfy_the_torch_minimum_the_probe_enforces():
    """The provisioner must not offer an image its own probe will reject.

    `runpod/pytorch:2.1.0-...` sat in IMAGES as the fallback, so whenever the preferred
    image was unavailable the loop created a pod from it and the probe then rejected that
    pod for shipping torch 2.1.0. Two of three rejections in one run were this, and it
    cost all twelve provisioning attempts. A minimum enforced on the host and violated in
    the configuration is not a minimum.
    """
    from tabicl.scaling import pod_runner

    pod_runner._check_images()          # the shipped list must pass

    original = pod_runner.IMAGES
    try:
        pod_runner.IMAGES = ["runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"]
        with pytest.raises(SystemExit, match="below the required"):
            pod_runner._check_images()
    finally:
        pod_runner.IMAGES = original


# --------------------------------------------------------------------------------------
# entity_label_history -- the entity's own earlier outcomes.
#
# The property under test is the one the docstring calls structural: a row cannot read its
# own label, because its own event becomes readable one horizon after its cutoff. These
# tests exist to make sure that stays true, since the gate that motivated this feature
# initially omitted the horizon and overstated it by 2.26 AUC.
# --------------------------------------------------------------------------------------

def _hist_frame():
    import pandas as pd
    t = pd.Timestamp("2020-01-01")
    # one entity with four monthly outcomes, one entity appearing exactly once
    ent = ["a", "a", "a", "a", "b"]
    times = [t, t + pd.Timedelta(days=30), t + pd.Timedelta(days=60),
             t + pd.Timedelta(days=90), t]
    vals = [1.0, 1.0, 0.0, 1.0, 1.0]
    return np.array(ent), np.array(vals), pd.to_datetime(pd.Series(times)).to_numpy()


def test_entity_label_history_never_reads_its_own_label():
    import pandas as pd
    from tabicl.scaling import entity_label_history
    ent, vals, times = _hist_frame()
    out = entity_label_history(ent, vals, times, ent, times,
                               label_horizon=pd.Timedelta(days=30))
    # The first row of each entity has no resolved predecessor at all.
    assert out["self__n_prior"].iloc[0] == 0
    assert np.isnan(out["self__positive_rate"].iloc[0])
    assert out["self__n_prior"].iloc[4] == 0          # entity 'b', single outcome
    # Row i of entity 'a' sees exactly its i predecessors, never itself.
    assert list(out["self__n_prior"].iloc[:4]) == [0, 1, 2, 3]
    # Rate at row 3 is over labels [1, 1, 0], not including its own 1.
    assert out["self__positive_rate"].iloc[3] == pytest.approx(2 / 3)


def test_entity_label_history_horizon_is_enforced_not_optional():
    import pandas as pd
    from tabicl.scaling import entity_label_history
    ent, vals, times = _hist_frame()
    for bad in (pd.Timedelta(0), pd.Timedelta(days=-1)):
        with pytest.raises(ValueError, match="must be positive"):
            entity_label_history(ent, vals, times, ent, times, label_horizon=bad)


def test_entity_label_history_horizon_delays_visibility():
    import pandas as pd
    from tabicl.scaling import entity_label_history
    ent, vals, times = _hist_frame()
    short = entity_label_history(ent, vals, times, ent, times,
                                 label_horizon=pd.Timedelta(days=30))
    long = entity_label_history(ent, vals, times, ent, times,
                                label_horizon=pd.Timedelta(days=75))
    # A longer horizon can only hide outcomes, never reveal them.
    assert (long["self__n_prior"] <= short["self__n_prior"]).all()
    assert long["self__n_prior"].sum() < short["self__n_prior"].sum()


def test_entity_label_history_unknown_entity_and_empty_pool():
    import pandas as pd
    from tabicl.scaling import entity_label_history
    ent, vals, times = _hist_frame()
    q_ent = np.array(["zz", "a"])
    q_t = pd.to_datetime(pd.Series([pd.Timestamp("2021-01-01")] * 2)).to_numpy()
    out = entity_label_history(ent, vals, times, q_ent, q_t,
                               label_horizon=pd.Timedelta(days=30))
    assert out["self__n_prior"].iloc[0] == 0            # never-seen entity
    assert np.isnan(out["self__positive_rate"].iloc[0])
    assert out["self__n_prior"].iloc[1] == 4            # 'a', all four resolved by then
    empty = entity_label_history(np.array([]), np.array([]),
                                 np.array([], dtype="datetime64[ns]"), q_ent, q_t,
                                 label_horizon=pd.Timedelta(days=30))
    assert len(empty) == 2 and empty["self__n_prior"].sum() == 0


def test_entity_label_history_preserves_query_order():
    import pandas as pd
    from tabicl.scaling import entity_label_history
    ent, vals, times = _hist_frame()
    order = [3, 0, 4, 2, 1]
    out = entity_label_history(ent, vals, times, ent[order], times[order],
                               label_horizon=pd.Timedelta(days=30))
    straight = entity_label_history(ent, vals, times, ent, times,
                                    label_horizon=pd.Timedelta(days=30))
    assert list(out["self__n_prior"]) == list(straight["self__n_prior"].iloc[order])


# --------------------------------------------------------------------------------------
# temporal_shift_grid -- the control's shift has to withhold something.
#
# rel-avito's grid degenerated to (0, 1) against a 4-day horizon: the first "control" was
# the unshifted setting and the second withheld nothing, yet the resulting 0.0053 difference
# was reported as a leak and excluded the whole +rate family from selection on that dataset.
# --------------------------------------------------------------------------------------

def test_temporal_shift_grid_never_returns_zero():
    from tabicl.scaling.eval_track_record import temporal_shift_grid
    # rel-avito: 8-day span. round(0.05*8)=0 and round(0.15*8)=1 under the old rule.
    for horizon in (4.0, 7.0):
        got = temporal_shift_grid(8.0, horizon)
        assert 0 not in got, got
        assert got, "a positive shift must exist"


def test_temporal_shift_grid_respects_the_label_horizon():
    from tabicl.scaling.eval_track_record import temporal_shift_grid
    # Nothing is withheld by a shift shorter than the horizon, so no shift may be.
    for span, horizon in ((8.0, 4.0), (8.0, 7.0), (3000.0, 365.0), (147.0, 7.0)):
        got = temporal_shift_grid(span, horizon)
        assert all(s >= horizon for s in got), (span, horizon, got)


def test_temporal_shift_grid_leaves_the_long_span_tasks_alone():
    from tabicl.scaling.eval_track_record import temporal_shift_grid
    # rel-event (147d span, 7d horizon) and rel-f1 already had valid grids; changing the
    # rule must not silently re-open verdicts that were reached under the old one.
    assert temporal_shift_grid(147.0, 7.0) == (7, 22)
    assert temporal_shift_grid(20000.0, 30.0) == (1000, 3000)


def test_temporal_shift_grid_deduplicates():
    from tabicl.scaling.eval_track_record import temporal_shift_grid
    # Both fractions clamp to the horizon on a short span: that is ONE control, not two.
    assert temporal_shift_grid(8.0, 4.0) == (4,)
    got = temporal_shift_grid(100.0, 10.0)
    assert len(got) == len(set(got))


def test_temporal_shift_grid_without_a_horizon_still_moves():
    from tabicl.scaling.eval_track_record import temporal_shift_grid
    # --no-horizon passes 0.0; a one-day floor still beats a zero shift.
    assert temporal_shift_grid(8.0, 0.0) == (1,)


# --------------------------------------------------------------------------------------
# abstention_choice -- keep the tuned pick only if its ranking survives a time gap.
#
# Motivated by rel-avito, where tuning loses on BOTH tasks and user-clicks drops 1.29 --
# four places in the published field -- because the arms are near-identical and selection
# is fitting noise on a small validation split.
# --------------------------------------------------------------------------------------

def test_abstention_falls_back_when_the_winner_loses_the_late_half():
    from tabicl.scaling.eval_track_record import abstention_choice
    split = {("+struct", 10000, "random"): (72.0, 68.0),    # wins early, loses late
             ("base", 10000, "random"):    (71.0, 70.0)}
    chosen, winner, abstained = abstention_choice(split)
    assert abstained is True
    assert winner[0] == "+struct"
    assert chosen[0] == "base"


def test_abstention_keeps_tuning_when_the_ranking_holds():
    from tabicl.scaling.eval_track_record import abstention_choice
    split = {("+struct", 10000, "random"): (72.0, 71.0),    # wins both halves
             ("base", 10000, "random"):    (71.0, 70.0)}
    chosen, winner, abstained = abstention_choice(split)
    assert abstained is False
    assert chosen == winner == ("+struct", 10000, "random")


def test_abstention_never_overrides_a_base_winner():
    from tabicl.scaling.eval_track_record import abstention_choice
    # base already won the early half; there is nothing to abstain from.
    split = {("+struct", 10000, "random"): (69.0, 60.0),
             ("base", 10000, "random"):    (71.0, 62.0)}
    chosen, winner, abstained = abstention_choice(split)
    assert abstained is False
    assert chosen[0] == "base" and winner[0] == "base"


def test_abstention_picks_the_best_base_configuration_not_just_any():
    from tabicl.scaling.eval_track_record import abstention_choice
    split = {("+struct", 10000, "random"): (75.0, 60.0),
             ("base", 1000, "random"):     (70.0, 65.0),
             ("base", 10000, "random"):    (74.0, 61.0)}
    chosen, _, abstained = abstention_choice(split)
    assert abstained is True
    # chosen on the EARLY half, like every other selection here -- not on the late one,
    # which is the judge and must not also be the chooser.
    assert chosen == ("base", 10000, "random")


def test_abstention_degrades_to_no_opinion_without_usable_entries():
    from tabicl.scaling.eval_track_record import abstention_choice
    nan = float("nan")
    assert abstention_choice({}) == (None, None, False)
    # all NaN (a validation half with one class) -> caller keeps its own argmax
    assert abstention_choice({("base", 10, "random"): (nan, nan)}) == (None, None, False)
    # no base arm at all -> nothing to fall back to
    assert abstention_choice({("+rate", 10, "random"): (70.0, 60.0)}) == (None, None, False)


# --------------------------------------------------------------------------------------
# two_hop_table -- depth-2 for repeated entity keys.
#
# rel-event and rel-avito were recorded as having no depth-2 available. They do: 25.8% and
# 23.8% of (grandchild, cutoff) pairs precede the cutoff. What blocked them was this
# package requiring unique entity keys for depth-2, not the data.
# --------------------------------------------------------------------------------------

def _two_hop_fixture():
    import pandas as pd
    t = pd.Timestamp("2021-01-01")
    child = pd.DataFrame({"cid": [1, 2, 3], "entity": ["a", "a", "b"],
                          "ctime": [t, t + pd.Timedelta(days=10), t]})
    grand = pd.DataFrame({
        "cid":  [1, 1, 2, 3],
        "gtime": [t, t + pd.Timedelta(days=5), t + pd.Timedelta(days=20), t],
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    return child, grand


def test_two_hop_table_keys_by_entity_and_keeps_the_grandchild_clock():
    from tabicl.scaling import two_hop_table, asof_statistics
    import pandas as pd
    child, grand = _two_hop_fixture()
    tbl = two_hop_table(grand, "cid", child, "cid", "entity", "g", time_column="gtime")
    assert tbl.foreign_key == "entity"
    assert tbl.time_column == "gtime"
    # join keys are structure, not signal, and must not reach the model
    assert "cid" not in tbl.df.columns
    # as-of at a cutoff between the rows sees only what precedes it
    out = asof_statistics(tbl, np.array(["a"]),
                          pd.to_datetime(pd.Series([pd.Timestamp("2021-01-08")])).to_numpy())
    assert out.filter(like="count").iloc[0, 0] == 2      # the two 'a' rows before day 8


def test_two_hop_table_respects_the_link_time_when_given():
    from tabicl.scaling import two_hop_table, asof_statistics
    import pandas as pd
    child, grand = _two_hop_fixture()
    # child 2 forms on day 10, so its grandchild (day 20) is unaffected; but a grandchild
    # that PRE-dates its own link must not become visible before the link exists.
    grand2 = grand.copy()
    grand2.loc[grand2["cid"] == 2, "gtime"] = pd.Timestamp("2021-01-02")
    without = two_hop_table(grand2, "cid", child, "cid", "entity", "g", time_column="gtime")
    with_link = two_hop_table(grand2, "cid", child, "cid", "entity", "g",
                              time_column="gtime", child_time_column="ctime")
    cut = pd.to_datetime(pd.Series([pd.Timestamp("2021-01-05")])).to_numpy()
    n_without = asof_statistics(without, np.array(["a"]), cut).filter(like="count").iloc[0, 0]
    n_with = asof_statistics(with_link, np.array(["a"]), cut).filter(like="count").iloc[0, 0]
    # as-of is STRICTLY before the cutoff, so the day-5 row is excluded from both.
    # Entity 'a' reaches grandchildren at days 0, 5 (via child 1) and 2 (via child 2).
    assert n_without == 2          # permissive: days 0 and 2; the link is assumed eternal
    assert n_with == 1             # correct: child 2 forms on day 10, so its day-2
                                   # grandchild is not visible at day 5 -- only day 0 remains


def test_two_hop_table_requires_a_grandchild_timestamp():
    from tabicl.scaling import two_hop_table
    child, grand = _two_hop_fixture()
    with pytest.raises(ValueError, match="grandchild's own timestamp"):
        two_hop_table(grand, "cid", child, "cid", "entity", "g", time_column="missing")


def test_two_hop_table_drops_unlinked_and_null_rows():
    from tabicl.scaling import two_hop_table
    import pandas as pd
    child, grand = _two_hop_fixture()
    orphan = pd.concat([grand, pd.DataFrame({"cid": [99], "gtime": [pd.NaT], "value": [9.0]})],
                       ignore_index=True)
    tbl = two_hop_table(orphan, "cid", child, "cid", "entity", "g", time_column="gtime")
    assert len(tbl.df) == len(grand)          # the orphan reaches no entity
    assert tbl.df["gtime"].notna().all()


def test_two_hop_table_survives_a_shared_column_name():
    """The child and grandchild both calling their timestamp the same thing.

    Not hypothetical: rel-event's `events` and `event_attendees` both use `start_time`.
    Merge suffixing then aliased the child's column onto the grandchild's, and dropping it
    destroyed the clock the table is built on -- a KeyError on the first real call, missed
    entirely by a fixture that used distinct names.
    """
    from tabicl.scaling import two_hop_table, asof_statistics
    import pandas as pd
    t = pd.Timestamp("2021-01-01")
    child = pd.DataFrame({"cid": [1, 2], "entity": ["a", "a"],
                          "ts": [t, t + pd.Timedelta(days=10)]})
    grand = pd.DataFrame({"cid": [1, 2], "ts": [t, t + pd.Timedelta(days=1)],
                          "value": [1.0, 2.0]})          # same name as the child's clock
    tbl = two_hop_table(grand, "cid", child, "cid", "entity", "g",
                        time_column="ts", child_time_column="ts")
    assert "ts" in tbl.df.columns and tbl.df["ts"].notna().all()
    assert "__link_time" not in tbl.df.columns and "__link_pk" not in tbl.df.columns
    # child 2 forms on day 10, so its grandchild is not visible at day 5; child 1's is.
    out = asof_statistics(tbl, np.array(["a"]),
                          pd.to_datetime(pd.Series([t + pd.Timedelta(days=5)])).to_numpy())
    assert out.filter(like="count").iloc[0, 0] == 1


def test_two_hop_table_dedupes_a_non_unique_link():
    """Reaching a SIBLING table through a shared parent, where the link repeats.

    rel-f1: drivers -> results -> constructorId -> constructor_results. The link
    `results[[driverId, constructorId]]` has one row per race, so without dedup each
    grandchild row is counted once per race the driver drove for that constructor -- every
    aggregate silently reweighted by how often the pair occurs.
    """
    from tabicl.scaling import two_hop_table, asof_statistics
    import pandas as pd
    t = pd.Timestamp("2021-01-01")
    # one driver, one constructor, THREE races -> the link repeats three times
    link = pd.DataFrame({"parent": [7, 7, 7], "entity": ["d", "d", "d"],
                         "ltime": [t, t + pd.Timedelta(days=1), t + pd.Timedelta(days=2)]})
    sib = pd.DataFrame({"parent": [7, 7], "stime": [t, t + pd.Timedelta(days=1)],
                        "pts": [10.0, 20.0]})
    tbl = two_hop_table(sib, "parent", link, "parent", "entity", "s",
                        time_column="stime", child_time_column="ltime")
    assert len(tbl.df) == 2, "each sibling row must appear once, not once per race"
    out = asof_statistics(tbl, np.array(["d"]),
                          pd.to_datetime(pd.Series([t + pd.Timedelta(days=10)])).to_numpy())
    assert out.filter(like="count").iloc[0, 0] == 2


# --------------------------------------------------------------------------------------
# paired.py -- stop computing paired deltas by hand over ssh.
# --------------------------------------------------------------------------------------

def test_paired_refuses_unequal_replicate_counts(tmp_path, capsys):
    from tabicl.scaling import paired
    # Truncating to the shorter run is the exact mistake this exists to prevent: a
    # 10-of-12 pairing nearly reported +0.63 where the 12-seed answer was +0.39, because
    # the two dropped seeds included the one bad draw.
    with pytest.raises(SystemExit, match="not paired runs"):
        paired.report([1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_reads_perseed_blocks_in_order(tmp_path):
    from tabicl.scaling import paired
    p = tmp_path / "run.log"
    p.write_text("noise\nPERSEED\t1.0\t2.0\nmore noise\nPERSEED\t3.0\t4.0\n", encoding="utf-8")
    assert paired.read_perseed(str(p)) == [[1.0, 2.0], [3.0, 4.0]]


def test_paired_reports_the_floor_verdict(capsys):
    from tabicl.scaling import paired
    paired.report([66.0] * 6, [66.3] * 6)          # +0.30, inside the floor
    assert "INSIDE the +-0.6 floor" in capsys.readouterr().out
    paired.report([66.0] * 6, [67.0] * 6)          # +1.00, outside it
    assert "above the +-0.6 floor" in capsys.readouterr().out


def test_paired_surfaces_a_variance_change_the_mean_hides():
    from tabicl.scaling import paired
    import io, contextlib
    # driver-dnf's real shape: mean barely moves, spread collapses.
    a = [70.02, 68.34, 65.60, 69.18, 71.34, 69.59, 71.17, 72.04]
    b = [71.20, 70.16, 69.76, 70.77, 69.84, 70.78, 70.03, 69.65]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        paired.report(a, b)
    out = buf.getvalue()
    assert "spread 2.05 -> 0.57" in out
    assert "worst seed 65.60 -> 69.65" in out


# --------------------------------------------------------------------------------------
# inference_config -- row chunking, the package's own scaling feature, which the benchmark
# runner had never used. Zero references to row_chunk or offload in eval_track_record until
# 2026-08-08, which is why rel-stack/user-engagement died at 88,137 test rows x 342 columns.
# --------------------------------------------------------------------------------------

def test_row_chunk_off_reproduces_the_original_config_exactly():
    from tabicl.scaling.eval_track_record import inference_config, NOAMP
    # Every standing number was measured on this exact dict. If "off" ever diverges from it,
    # a rerun silently stops reproducing the table.
    assert inference_config("off") == NOAMP
    assert "row_chunk" not in inference_config("off")["COL_CONFIG"]


def test_row_chunk_modes_set_the_documented_values():
    from tabicl.scaling.eval_track_record import inference_config
    assert inference_config("auto")["COL_CONFIG"]["row_chunk"] == "auto"
    assert inference_config("always")["COL_CONFIG"]["row_chunk"] is True


def test_row_chunk_never_mutates_the_shared_default():
    from tabicl.scaling.eval_track_record import inference_config, NOAMP
    # A shallow copy here would let one call leak chunking into every later one, including
    # the arm it is being compared against -- an A/B where both arms silently became B.
    inference_config("always")
    inference_config("auto")
    assert "row_chunk" not in NOAMP["COL_CONFIG"]


def test_row_chunk_leaves_amp_off_in_every_mode():
    from tabicl.scaling.eval_track_record import inference_config
    # AMP costs 7.3 AUC on rel-event. It must stay off whatever chunking does.
    for mode in ("off", "auto", "always"):
        cfg = inference_config(mode)
        assert all(cfg[k]["use_amp"] is False for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"))


# --- grid_noise: the diagnostic that says whether calibrated selection means anything ----
# These target how it could report a confident WRONG number, not whether it runs. Its whole
# output is a ratio, and both halves of that ratio are easy to corrupt silently.

_GRID_LOG = """\
  base                 context=1000   val=60.00
  +struct              context=1000   val=62.00
  chosen +struct context=1000 order=random -> VAL 62.00  TEST 55.00
  base                 context=1000   val=61.00
  +struct              context=1000   val=63.00
  chosen +struct context=1000 order=random -> VAL 63.00  TEST 56.00
  base                 context=1000   val=62.00
  +struct              context=1000   val=64.00
  chosen +struct context=1000 order=random -> VAL 64.00  TEST 57.00
rel-x/task-a  CALIBRATED TEST ROC-AUC x100 = 56.00 +- 1.00 over 3 replicates
  base                 context=1000   val=70.00
  +struct              context=1000   val=71.00
  chosen +struct context=1000 order=random -> VAL 71.00  TEST 65.00
rel-x/task-b  CALIBRATED TEST ROC-AUC x100 = 65.00 +- 0.00 over 1 replicates
"""


def test_grid_noise_does_not_let_two_blocks_run_together():
    from tabicl.scaling.grid_noise import parse_blocks
    # The sibling parser reported +17.66 on a two-point-range task by flushing only at arm
    # headers. Here the same bug would pool task-a's 60s with task-b's 70s and report a
    # noise of ~4 instead of ~1, turning a lottery verdict into a decisive one.
    blocks = parse_blocks(_GRID_LOG)
    assert [b[0] for b in blocks] == ["rel-x/task-a", "rel-x/task-b"]
    assert [len(b[1]) for b in blocks] == [3, 1]


def test_grid_noise_excludes_the_chosen_line_from_the_grid():
    from tabicl.scaling.grid_noise import parse_blocks
    # `chosen` repeats the winner's score. Counting it as a candidate would make the top two
    # entries identical and drive every margin to 0.00 -- a spurious unanimous "lottery".
    cands = parse_blocks(_GRID_LOG)[0][1][0][0]
    assert set(cands) == {("base", 1000), ("+struct", 1000)}


def test_grid_noise_computes_margin_and_noise_on_a_known_case():
    from tabicl.scaling.grid_noise import parse_blocks, block_stats
    st = block_stats(parse_blocks(_GRID_LOG)[0][1])
    assert st["margin"] == pytest.approx(2.0)      # 62-60, 63-61, 64-62
    assert st["noise"] == pytest.approx(1.0)       # sd(60,61,62) == sd(62,63,64) == 1.0
    assert st["ratio"] == pytest.approx(2.0)
    assert st["arm_stability"] == 1.0 and st["n_arms"] == 1


def test_grid_noise_refuses_a_noise_estimate_from_two_seeds():
    from tabicl.scaling.grid_noise import parse_blocks, block_stats
    # sd of two numbers is not a noise estimate, and this module's headline IS a ratio whose
    # denominator is that sd. Returning None beats returning a confident quotient.
    assert block_stats(parse_blocks(_GRID_LOG)[1][1]) is None


def test_grid_noise_intersects_candidates_across_seeds():
    from tabicl.scaling.grid_noise import parse_blocks, block_stats
    # A candidate only some seeds scored cannot be a runner-up: comparing the winner against
    # something one seed never measured invents a margin. Seed 3 drops +struct, so the
    # intersection has a single candidate and the block declines to report.
    log = _GRID_LOG.replace("  +struct              context=1000   val=64.00\n", "")
    assert block_stats(parse_blocks(log)[0][1]) is None


def test_offload_off_reproduces_the_original_config_exactly():
    from tabicl.scaling.eval_track_record import inference_config, NOAMP
    # Same contract as --row-chunk off: every standing number was measured on this dict.
    assert inference_config("off", "off") == NOAMP
    assert inference_config("off") == NOAMP


def test_offload_sets_every_stage_not_just_the_column_one():
    from tabicl.scaling.eval_track_record import inference_config
    # The point of the flag is the ICL stage. Out of the box COL_CONFIG.offload is already
    # "auto" while ICL_CONFIG.offload is False, so a version of this that set only
    # COL_CONFIG would change nothing and read as "offloading did not help".
    cfg = inference_config("off", "cpu")
    assert all(cfg[k]["offload"] == "cpu"
               for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"))


def test_offload_and_row_chunk_compose_without_clobbering():
    from tabicl.scaling.eval_track_record import inference_config
    # They address different tensors -- activations vs outputs -- so a shape can need both.
    cfg = inference_config("auto", "cpu")
    assert cfg["COL_CONFIG"]["row_chunk"] == "auto"
    assert cfg["COL_CONFIG"]["offload"] == "cpu"


def test_offload_never_mutates_the_shared_default():
    from tabicl.scaling.eval_track_record import inference_config, NOAMP
    inference_config("auto", "cpu")
    inference_config("always", "disk")
    assert NOAMP == {k: {"use_amp": False}
                     for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}


@pytest.mark.parametrize("row_chunk", ["off", "auto", "always"])
@pytest.mark.parametrize("offload", ["off", "auto", "cpu", "disk"])
def test_every_flag_combination_is_accepted_by_tabicl(row_chunk, offload):
    from tabicl._model.inference_config import InferenceConfig
    from tabicl.scaling.eval_track_record import inference_config
    # The runner builds this dict and hands it to TabICLClassifier. A key or value the
    # library rejects turns into a crash hours into a pod run, after feature building has
    # already been paid for -- which is the expensive way to learn that MgrConfig is the
    # TYPE of the stage configs rather than a fourth key beside them.
    cfg = InferenceConfig()
    cfg.update_from_dict(inference_config(row_chunk, offload))
    assert all(getattr(cfg, k).use_amp is False
               for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"))


# --- rank_table: the headline table's arithmetic, which has been wrong by hand before ----

def test_rank_table_reproduces_every_published_cell():
    from tabicl.scaling.rank_table import rank, OURS
    # The rule was inferred from these seven, so this is the test that keeps it inferred.
    # DFS is displayed in the table but NOT ranked, which is what makes the denominator 10.
    published = {"rel-event/user-repeat": 3, "rel-trial/study-outcome": 4,
                 "rel-f1/driver-top3": 6, "rel-event/user-ignore": 7,
                 "rel-avito/user-visits": 8, "rel-avito/user-clicks": 8,
                 "rel-f1/driver-dnf": 9}
    for task, expected in published.items():
        r, n = rank(OURS[task], task)
        assert (r, n) == (expected, 10), f"{task}: got {r}/{n}, table says {expected}/10"


def test_rank_table_has_all_twelve_tasks_and_nine_methods():
    from tabicl.scaling.rank_table import FIELD, METHODS
    assert len(FIELD) == 12 and len(METHODS) == 9
    assert all(len(v) == 9 for v in FIELD.values())


def test_rank_refuses_a_task_with_no_published_field():
    from tabicl.scaling.rank_table import rank
    # Inventing a denominator would produce a rank that looks authoritative and means
    # nothing. Better to fail than to publish it.
    with pytest.raises(KeyError):
        rank(70.0, "rel-nonesuch/made-up")


SEVEN = {"rel-event/user-repeat": 77.89, "rel-trial/study-outcome": 72.26,
         "rel-f1/driver-top3": 81.98, "rel-event/user-ignore": 80.98,
         "rel-avito/user-visits": 65.54, "rel-avito/user-clicks": 65.89,
         "rel-f1/driver-dnf": 69.66}


def test_average_is_a_rank_of_means_not_a_mean_of_ranks():
    from tabicl.scaling.rank_table import average_rank, averages
    # Pinned to the SEVEN tasks explicitly, not to whatever OURS currently holds. OURS grows
    # as tasks land -- rel-hm/user-churn joined on 2026-08-08 and moved these to 72.62 and
    # 7/10 -- and a test that follows it would stop checking anything.
    ar, an = average_rank(list(SEVEN), SEVEN)
    assert (ar, an) == (6, 10)          # the published seven-task cell
    assert isinstance(ar, int)          # "6.43" was a mean of ranks and ranked us against nothing
    assert averages(list(SEVEN), SEVEN)["ours"] == pytest.approx(73.46, abs=0.005)


def test_the_added_tasks_cost_us_three_places():
    from tabicl.scaling.rank_table import average_rank, averages, OURS
    # All three were run because they were missing, not because they looked winnable. They
    # drop us 6th -> 9th. Pinned so the cost cannot be quietly undone by a later edit.
    assert len(OURS) == 12
    assert average_rank(list(OURS)) == (9, 10)
    assert averages(list(OURS))["ours"] == pytest.approx(75.10, abs=0.005)


def test_a_rising_average_hid_a_falling_rank():
    from tabicl.scaling.rank_table import averages, OURS
    # The trap this table is now built to expose: our mean went UP by 0.26 while our
    # position went DOWN three places, because every rival gained more from the added
    # tasks than we did. A reader seeing only "73.46 -> 73.72" would conclude we improved.
    seven = {k: v for k, v in OURS.items()
             if k not in ("rel-hm/user-churn", "rel-stack/user-engagement",
                          "rel-amazon/user-churn", "rel-amazon/item-churn",
                          "rel-stack/user-badge")}
    before, after = averages(list(seven), seven), averages(list(OURS))
    assert after["ours"] > before["ours"]
    assert after["GraphSAGE"] - before["GraphSAGE"] > after["ours"] - before["ours"]


def test_partial_seed_counts_are_recorded_for_the_table_to_mark():
    from tabicl.scaling.rank_table import PARTIAL_SEEDS, OURS
    # rel-stack/user-engagement is four replicates, salvaged from a timed-out run. A real
    # measurement and a weaker one; the table must be able to mark it rather than present
    # it as the standard five.
    assert PARTIAL_SEEDS["rel-stack/user-engagement"] == 4
    assert all(k in OURS for k in PARTIAL_SEEDS)


def test_average_refuses_a_task_set_we_have_not_measured():
    from tabicl.scaling.rank_table import averages
    # Averaging every method over 12 tasks while averaging ourselves over 7 is precisely the
    # flattery this table was corrected for once already.
    # Every task is measured now, so the refusal is exercised with an explicit `ours` that
    # is missing one -- which is the real situation it guards: averaging every method over
    # twelve tasks while averaging ourselves over fewer.
    partial = {"rel-f1/driver-dnf": 69.66}
    with pytest.raises(ValueError):
        averages(["rel-stack/user-badge", "rel-f1/driver-dnf"], partial)


def test_adding_a_task_changes_the_average_row_for_everyone():
    from tabicl.scaling.rank_table import averages, OURS
    # The five missing tasks are not a random sample of difficulty -- rel-stack sits at
    # 88-91. Adding one lifts EVERY method's average, so the row must never be compared
    # across two different task sets.
    base = averages(list(OURS))
    wider = averages(list(OURS) + ["rel-stack/user-badge"],
                     {**OURS, "rel-stack/user-badge": 80.0})
    assert all(wider[m] > base[m] for m in ("RelGNN", "GraphSAGE", "TabPFN-REL"))


# --- cgroup-aware CPU memory: the bug behind two silent multi-hour failures ----------
# psutil.virtual_memory().available reports the HOST's free memory and is blind to cgroup
# limits. Measured on a RunPod L40S container: 450.8 GB reported against 102.4 GB real, a
# 4.4x overestimate. It decides whether offload="auto" escalates to disk, so overestimating
# means never escalating -- and the cgroup OOM killer sends SIGKILL, so the process dies with
# no traceback and no CUDA error. Indistinguishable from a clean exit.

def test_cgroup_headroom_is_none_without_a_limit(tmp_path, monkeypatch):
    from tabicl._model import inference
    # No cgroup files (Windows, bare metal) -> fall back to psutil unchanged.
    monkeypatch.chdir(tmp_path)
    assert inference._cgroup_memory_headroom() is None or isinstance(
        inference._cgroup_memory_headroom(), int)


def test_cgroup_headroom_reads_v2_and_subtracts_usage(tmp_path, monkeypatch):
    import pathlib
    from tabicl._model import inference
    real = pathlib.Path.read_text

    def fake(self, *a, **k):
        # Deliberately below any plausible host RAM: the helper treats a limit at or
        # above physical memory as "no real constraint" (that is how cgroup v1's
        # unlimited sentinel is caught), so a 125 GB fixture would be discarded on a
        # 64 GB test machine and the assertion would test nothing.
        if str(self).endswith("memory.max"):
            return "8000000000"
        if str(self).endswith("memory.current"):
            return "1000000000"
        raise OSError("no such file")

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    monkeypatch.setattr(pathlib.Path, "read_text", fake)
    try:
        assert inference._cgroup_memory_headroom() == 8000000000 - 1000000000
    finally:
        monkeypatch.setattr(pathlib.Path, "read_text", real)


def test_cgroup_headroom_treats_max_as_unlimited(monkeypatch):
    import pathlib
    from tabicl._model import inference
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: "max")
    assert inference._cgroup_memory_headroom() is None


def test_cgroup_headroom_ignores_the_v1_unlimited_sentinel(monkeypatch):
    import pathlib
    from tabicl._model import inference
    # cgroup v1 signals "unlimited" with a value near 2**63. Treating that as a real limit
    # would make every uncontained run think it had exabytes of headroom.
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: str(9223372036854771712))
    assert inference._cgroup_memory_headroom() is None


def test_cgroup_headroom_never_raises(monkeypatch):
    import pathlib
    from tabicl._model import inference
    # A memory estimate is not worth crashing an inference call over. Every failure path
    # must fall back to the previous behaviour.
    def boom(self, *a, **k):
        raise OSError("permission denied")
    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    assert inference._cgroup_memory_headroom() is None


def test_available_cpu_memory_can_only_shrink(monkeypatch):
    import psutil
    from tabicl._model import inference
    # The cgroup figure is INTERSECTED with psutil's, never substituted, so this change can
    # only ever be more conservative than the behaviour every existing result was measured
    # under.
    mgr = inference.InferenceManager.__new__(inference.InferenceManager)
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: 1024 ** 3)
    got = inference.InferenceManager.get_available_cpu_memory(mgr)
    assert got == pytest.approx(1024.0)                      # 1 GiB in MB, the cgroup cap
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: None)
    got = inference.InferenceManager.get_available_cpu_memory(mgr)
    assert got == pytest.approx(psutil.virtual_memory().available / 1024 ** 2, rel=0.2)


# --- salvage: a timed-out run is a weaker measurement, not a lost one ------------------
# rel-stack/user-engagement costs ~55 min/seed once disk offloading engages, against a 3h
# ceiling, so the SUMMARY line is the part most likely to be missing -- on exactly the tasks
# that were hardest to measure at all. These tests are about the one way this goes wrong
# quietly: a partial mean being read as a complete one.

_PARTIAL_LOG = """\
########## NEW TASK :: rel-stack/user-engagement :: (01:00:00Z)
  chosen +history context=10000 order=random -> VAL 89.53  TEST 89.34
  chosen +counts context=10000 order=random -> VAL 89.11  TEST 89.12
EXIT rel-stack/user-engagement rc=124 TIMED OUT after 10800s -- a missing measurement
########## NEW TASK :: rel-hm/user-churn :: (04:00:00Z)
  chosen base context=5000 order=random -> VAL 70.10  TEST 66.50
  chosen base context=5000 order=random -> VAL 70.20  TEST 67.00
rel-hm/user-churn  CALIBRATED TEST ROC-AUC x100 = 66.75 +- 0.23 over 2 replicates
"""


def test_salvage_recovers_seeds_from_a_run_with_no_summary():
    from tabicl.scaling.salvage import salvage
    r = {x.task: x for x in salvage(_PARTIAL_LOG)}
    eng = r["rel-stack/user-engagement"]
    assert eng.tests == [89.34, 89.12]
    assert eng.mean == pytest.approx(89.23, abs=0.005)
    assert eng.complete is False and "rc=124" in eng.exit_note


def test_salvage_marks_a_complete_run_complete():
    from tabicl.scaling.salvage import salvage
    r = {x.task: x for x in salvage(_PARTIAL_LOG)}
    assert r["rel-hm/user-churn"].complete is True
    assert r["rel-hm/user-churn"].exit_note == ""


def test_partial_results_always_print_their_seed_count_and_a_marker():
    from tabicl.scaling.salvage import salvage, format_result
    # The specific failure this guards: a 2-seed mean transcribed into the headline table
    # as though it were the standard five.
    r = {x.task: x for x in salvage(_PARTIAL_LOG)}
    line = format_result(r["rel-stack/user-engagement"])
    assert "PARTIAL" in line and "2 seeds" in line
    assert "PARTIAL" not in format_result(r["rel-hm/user-churn"])


def test_salvage_does_not_pool_seeds_across_tasks():
    from tabicl.scaling.salvage import salvage
    # Letting one task's seeds run into the next is how a sibling parser in this project
    # reported +17.66 on a task with a two-point range.
    r = {x.task: x for x in salvage(_PARTIAL_LOG)}
    assert r["rel-hm/user-churn"].tests == [66.50, 67.00]
    assert 89.34 not in r["rel-hm/user-churn"].tests


def test_a_task_that_produced_nothing_is_reported_as_missing_not_as_zero():
    from tabicl.scaling.salvage import salvage, format_result
    log = ("########## NEW TASK :: rel-stack/user-badge :: (01:00:00Z)\n"
           "EXIT rel-stack/user-badge rc=137 SIGKILL (9) -- almost certainly out of memory\n")
    (r,) = salvage(log)
    assert r.tests == []
    assert "NO SEEDS COMPLETED" in format_result(r)
    assert "missing measurement" in format_result(r)


# --- --train-pool: release train rows the fit will never use --------------------------
# The fit draws --context rows (10,000). Every train row is materialised for every arm
# anyway: rel-stack/user-badge builds ~48 GB (5 arms x 3.39M rows x ~350 cols x 8B) to use
# 0.3% of it, then dies when inference buffers land on top. That it is arms x rows x columns
# rather than rows alone is settled by measurement -- rel-amazon/user-churn survived 4.7M
# train rows and 352k test rows on a single 52-column arm.

def _pool_args(**kw):
    import argparse
    d = dict(train_pool=0, context=10000, context_orders="random")
    d.update(kw)
    return argparse.Namespace(**d)


def test_train_pool_default_is_off_so_standing_numbers_reproduce():
    import pathlib as _p
    # Every standing number was measured with the whole train pool resident. The flag must
    # default to 0 so a rerun reproduces them; a default that quietly subsampled the fit
    # pool would change results without changing the command line.
    src = _p.Path("src/tabicl/scaling/eval_track_record.py").read_text(encoding="utf-8")
    i = src.index('"--train-pool"')
    assert "default=0" in src[i:i + 200]


def test_train_pool_subsample_keeps_arms_labels_and_frame_aligned():
    import numpy as np
    import pandas as pd
    # The alignment this protects: `time_order` and the gap-validation pools are derived
    # from `train` further down, so subsampling the arms without subsampling `train` and
    # `y` misaligns every index into the fit pool -- silently, and in a way that would read
    # as a modelling result rather than a bug.
    n, pool = 1000, 200
    arms = {"base": (np.arange(n * 3).reshape(n, 3), np.zeros((10, 3)))}
    y = np.arange(n)
    train = pd.DataFrame({"t": np.arange(n)})
    keep = np.sort(np.random.default_rng(0).choice(n, size=pool, replace=False))
    arms = {k: (v[0][keep], v[1]) for k, v in arms.items()}
    y2, train2 = y[keep], train.iloc[keep].reset_index(drop=True)
    assert len(arms["base"][0]) == len(y2) == len(train2) == pool
    # row i of every subsampled object must still describe the same original row
    assert arms["base"][0][0, 0] == keep[0] * 3
    assert y2[0] == keep[0] and train2["t"].iloc[0] == keep[0]
    assert arms["base"][1].shape[0] == 10          # the eval matrix is untouched


def test_train_pool_is_a_uniform_draw_and_deterministic():
    import numpy as np
    a = np.sort(np.random.default_rng(0).choice(1000, size=200, replace=False))
    b = np.sort(np.random.default_rng(0).choice(1000, size=200, replace=False))
    assert (a == b).all()                          # seed 0, so two runs agree
    assert len(set(a.tolist())) == 200             # without replacement


def test_grid_noise_flushes_on_a_task_header_not_only_on_a_summary():
    from tabicl.scaling.grid_noise import parse_blocks, block_stats
    # A task killed by a timeout or the OOM killer never prints a CALIBRATED summary, so a
    # parser that flushes only on summaries lets its seeds run into the NEXT task. That is
    # what happened on 2026-08-09: rel-stack/user-engagement timed out after four seeds and
    # its 89.33s pooled with rel-amazon/user-churn's 66.94s, producing one 9-seed block at
    # "noise" 11.38 on tasks whose real sds are 0.20 and 0.15. The published block count was
    # wrong (34 rather than 41) for the same reason.
    log = """\
########## NEW TASK :: rel-a/dead ::
  base                 context=1000   val=89.00
  chosen base context=1000 order=random -> VAL 89.00  TEST 89.33
  base                 context=1000   val=89.10
  chosen base context=1000 order=random -> VAL 89.10  TEST 89.20
########## NEW TASK :: rel-b/alive ::
  base                 context=1000   val=70.00
  +x                   context=1000   val=71.00
  chosen +x context=1000 order=random -> VAL 71.00  TEST 66.90
  base                 context=1000   val=70.10
  +x                   context=1000   val=71.20
  chosen +x context=1000 order=random -> VAL 71.20  TEST 67.00
  base                 context=1000   val=70.20
  +x                   context=1000   val=71.10
  chosen +x context=1000 order=random -> VAL 71.10  TEST 66.95
rel-b/alive  CALIBRATED TEST ROC-AUC x100 = 66.95 +- 0.05 over 3 replicates
"""
    blocks = parse_blocks(log)
    assert [b[0] for b in blocks] == ["rel-b/alive"]      # the dead task is discarded
    (_, seeds, _), = blocks
    assert len(seeds) == 3                                 # not 5
    st = block_stats(seeds)
    assert st["noise"] < 1.0                               # not 11.38


def test_numeric_drops_array_valued_columns_instead_of_crashing():
    import numpy as np
    import pandas as pd
    from tabicl.scaling.eval_track_record import _numeric
    # rel-amazon carries a column of per-row vectors. pd.factorize raises
    # "TypeError: unhashable type: 'numpy.ndarray'" on it, which killed
    # rel-amazon/item-churn AFTER its features were built and its controls had run.
    df = pd.DataFrame({
        "n": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "a"],
        "emb": [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])],
    })
    X = _numeric(df, fit=True, tag="arrtest")
    assert X.shape == (3, 2)                      # emb dropped, n and cat kept
    assert np.isfinite(X).all()


def test_numeric_does_not_stringify_array_columns_into_row_identifiers():
    import numpy as np
    import pandas as pd
    from tabicl.scaling.eval_track_record import _numeric
    # Stringifying would mint one category per row -- a column of distinct codes carrying
    # no signal and indistinguishable from a row identifier, which is the exact shape the
    # leak controls exist to catch. Dropping must leave no column with all-distinct codes.
    df = pd.DataFrame({"emb": [np.array([float(i)]) for i in range(5)]})
    X = _numeric(df, fit=True, tag="arronly")
    assert X.shape[1] == 0


def test_numeric_still_encodes_ordinary_categoricals():
    import numpy as np
    import pandas as pd
    from tabicl.scaling.eval_track_record import _numeric
    df = pd.DataFrame({"cat": ["x", "y", "x", "z"]})
    X = _numeric(df, fit=True, tag="plaincat")
    assert X.shape == (4, 1)
    assert X[0, 0] == X[2, 0] and X[0, 0] != X[1, 0]


# --- the temporal control must compare INFORMATION, not raw score ----------------------
# Controls 3 and 4 score a single summed column directly against the label -- no model is
# fitted -- so a value below chance is an INVERTED feature, not a weak one. On rel-amazon,
# items with more reviews are less likely to churn: n_linked scores 0.32, which carries what
# 0.68 carries. Comparing raw scores inverts the verdict there, and did: six LEAK verdicts
# were issued below chance against seven genuine ones above it.

def test_temporal_control_passes_an_inverted_but_informative_block():
    from tabicl.scaling._leakage import temporal_control
    # rel-amazon/item-churn's real numbers. Raw AUC rises 0.3236 -> 0.3388 and the old code
    # called it a leak; the deviations FALL 0.1764 -> 0.1612, so withholding history reduced
    # the information, which is the opposite of a leak.
    r = temporal_control(lambda s: {0.0: 0.3236, 136.0: 0.3330, 410.0: 0.3388}[s],
                         shifts=(0.0, 136.0, 410.0))
    assert r.passed is True and r.inconclusive is False
    assert "do not add information" in r.reason


def test_every_real_below_chance_verdict_flips_to_pass():
    from tabicl.scaling._leakage import temporal_control
    # All six misfires recorded across this project's runs, with their real shifted values.
    for obs, shifted in ((0.3236, 0.3388), (0.3207, 0.3369), (0.4039, 0.4232),
                         (0.4090, 0.4266), (0.4186, 0.4256), (0.4671, 0.4854)):
        r = temporal_control(lambda s, o=obs, w=shifted: {0.0: o, 30.0: w}[s],
                             shifts=(0.0, 30.0))
        assert r.passed, f"{obs} -> {shifted} should pass on |score - chance|"


def test_temporal_control_still_catches_a_real_leak_above_chance():
    from tabicl.scaling._leakage import temporal_control
    r = temporal_control(lambda s: {0.0: 0.70, 30.0: 0.78}[s], shifts=(0.0, 30.0))
    assert r.passed is False and r.inconclusive is False
    assert "reaching past the cutoff" in r.reason


def test_temporal_control_still_passes_a_clean_block():
    from tabicl.scaling._leakage import temporal_control
    r = temporal_control(lambda s: {0.0: 0.70, 30.0: 0.66}[s], shifts=(0.0, 30.0))
    assert r.passed is True and r.inconclusive is False


def test_above_chance_verdicts_are_identical_under_both_measures():
    from tabicl.scaling._leakage import temporal_control
    # The fix must not disturb any verdict that was already correct: above chance,
    # |score - chance| and the raw score order the same way.
    for obs, shifted in ((0.5618, 0.5700), (0.6926, 0.7100), (0.8179, 0.8300),
                         (0.7000, 0.6600), (0.6155, 0.6000)):
        r = temporal_control(lambda s, o=obs, w=shifted: {0.0: o, 30.0: w}[s],
                             shifts=(0.0, 30.0))
        assert r.passed == (shifted <= obs + 0.005)


def test_inconclusive_only_when_both_sides_are_at_chance():
    from tabicl.scaling._leakage import temporal_control
    # The one case with genuinely nothing to compare. Narrow on purpose: an inverted block
    # is informative and must get a real verdict, not an abstention.
    r = temporal_control(lambda s: {0.0: 0.501, 30.0: 0.502}[s], shifts=(0.0, 30.0))
    assert r.inconclusive is True
    assert "no information on either side" in r.reason


def test_permutation_control_is_two_sided():
    import numpy as np
    from tabicl.scaling import permutation_control
    # A purely label-derived block scoring 0.4415 after its labels are shuffled carries
    # 0.0585 of information about the true label, inverted, and the one-sided
    # `mean <= chance + tolerance` read 0.4415 <= 0.52 and passed it.
    #
    # NOT a real case from this benchmark: eval_track_record calls permutation_test, never
    # this function, so no result here was ever gated by it. 0.4415 was originally cited as
    # rel-avito/user-clicks evidence and that was wrong twice over -- it is
    # permutation_test's null, and that block contains count columns which survive label
    # shuffling by construction, so a null away from chance is expected there.
    y = np.array([0, 1] * 50)
    r = permutation_control(lambda labels: 0.4415, y, n_permutations=3)
    assert r.passed is False
    assert "inverted" in r.reason


def test_permutation_control_still_passes_a_clean_null():
    import numpy as np
    from tabicl.scaling import permutation_control
    y = np.array([0, 1] * 50)
    assert permutation_control(lambda labels: 0.4999, y, n_permutations=3).passed
    assert permutation_control(lambda labels: 0.5057, y, n_permutations=3).passed


def test_permutation_control_still_catches_the_above_chance_case():
    import numpy as np
    from tabicl.scaling import permutation_control
    y = np.array([0, 1] * 50)
    r = permutation_control(lambda labels: 0.5510, y, n_permutations=3)
    assert r.passed is False and "inverted" not in r.reason


def test_the_two_control_changes_pull_in_opposite_directions():
    import numpy as np
    from tabicl.scaling import permutation_control, temporal_control
    # Comparing information rather than raw score makes the temporal control MORE permissive
    # on an inverted block and the permutation control STRICTER on one. Only the first is
    # reachable from eval_track_record; the second is a library-facing fix with no effect on
    # any benchmark number here.
    y = np.array([0, 1] * 50)
    assert temporal_control(lambda s: {0.0: 0.3236, 30.0: 0.3388}[s],
                            shifts=(0.0, 30.0)).passed is True     # was LEAK
    assert permutation_control(lambda labels: 0.4415, y, n_permutations=3).passed is False  # was PASS


# --- as-of aggregation: child rows for unqueried keys are dead weight -------------------
# The prefix arrays are built over the FULL child table -- three float64 columns plus a
# count over len(df) -- while every lookup is confined to one key's block. rel-stack's child
# tables run to millions of rows spanning every user in the database, which is why
# user-badge failed on memory under --row-chunk, --offload and --train-pool alike: all of
# those bound the model or the fit pool, none bound the aggregation.

def _asof_frame():
    import numpy as np
    import pandas as pd
    from tabicl.scaling import Table
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "uid": rng.integers(0, 40, n),
        "ts": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 100, n), "D"),
        "v": rng.normal(1000.0, 5.0, n),
    })
    return Table(df, foreign_key="uid", name="ev", time_column="ts")


def test_asof_is_unchanged_by_dropping_unqueried_keys():
    import numpy as np
    import pandas as pd
    from tabicl.scaling import asof_statistics
    # The exactness claim, tested directly: aggregating over the full child table and over
    # a child table pre-filtered to the queried keys must agree, because rows of other keys
    # live only in blocks that are never indexed.
    tbl = _asof_frame()
    keys = np.array([1, 2, 3, 5, 8])
    cutoffs = np.array([pd.Timestamp("2020-03-01")] * len(keys), dtype="datetime64[ns]")
    full = asof_statistics(tbl, keys, cutoffs)

    from dataclasses import replace
    pre = replace(tbl, df=tbl.df[tbl.df["uid"].isin(keys)].reset_index(drop=True))
    trimmed = asof_statistics(pre, keys, cutoffs)

    assert list(full.columns) == list(trimmed.columns)
    pd.testing.assert_frame_equal(full.reset_index(drop=True),
                                  trimmed.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9, atol=1e-9)


def test_asof_still_correct_when_a_key_has_no_child_rows():
    import numpy as np
    import pandas as pd
    from tabicl.scaling import asof_statistics
    # The filter must not turn "key present but no rows before the cutoff" into a missing
    # row: a key with nothing to aggregate still needs an output row.
    tbl = _asof_frame()
    keys = np.array([1, 9999])            # 9999 appears in no child row at all
    cutoffs = np.array([pd.Timestamp("2020-03-01")] * 2, dtype="datetime64[ns]")
    out = asof_statistics(tbl, keys, cutoffs)
    assert len(out) == 2


def test_asof_handles_an_entirely_disjoint_key_set():
    import numpy as np
    import pandas as pd
    from tabicl.scaling import asof_statistics
    # Every child row filtered away. Must still return one row per query rather than raise.
    tbl = _asof_frame()
    keys = np.array([5000, 5001])
    cutoffs = np.array([pd.Timestamp("2020-03-01")] * 2, dtype="datetime64[ns]")
    out = asof_statistics(tbl, keys, cutoffs)
    assert len(out) == 2



def test_the_twelve_task_average_reproduces_the_published_column():
    from tabicl.scaling.rank_table import averages, OURS, FIELD
    # The decisive cross-check on every field figure transcribed into this module: with all
    # twelve tasks present, our computed averages must equal Table 14's own Avg AUROC column
    # exactly. If a single cell were mistyped this would not hold.
    published = {"RelGNN": 78.06, "RelGT": 76.65, "GraphSAGE": 75.83, "Griffin": 66.29,
                 "RDBLearn": 75.55, "RDBLearn+v2.5": 75.31, "RDBLearn+v3": 76.01,
                 "KumoRFMv2": 75.91, "TabPFN-REL": 76.91}
    assert len(OURS) == len(FIELD) == 12
    got = averages(list(FIELD))
    for method, expected in published.items():
        assert got[method] == pytest.approx(expected, abs=0.005), method


# --- NFA: the one traversal shape this package never had -------------------------------
# Every existing traversal leaves the entity table through a foreign key. This one links
# entity rows to EACH OTHER by shared attribute values -- the incidence graph of Cucumides &
# Geerts, "Grables: Tabular Learning Beyond Independent Rows" (arXiv 2602.03945). On
# relbench-trial their fixed within-table aggregations (0.7254) beat row-local LightGBM
# (0.7009), every GNN baseline (0.6860, 0.6861) and learned message passing over the
# cross-table structure we already build (0.7180).

def _nfa_frame(n=200, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sponsor": rng.choice([f"s{i}" for i in range(10)], n),
        "unique_id": [f"u{i}" for i in range(n)],
        "constant": ["x"] * n,
        "ts": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
        "value": rng.normal(50.0, 10.0, n),
    })


def test_nfa_columns_rejects_near_unique_and_near_constant():
    from tabicl.scaling._relational import nfa_columns
    # A near-unique column gives every row its own group and aggregates over nothing; a
    # near-constant one gives a single global group and the aggregate degenerates to a
    # dataset mean. Both are useless and both are cheap to exclude without labels.
    cols = nfa_columns(_nfa_frame(), exclude=["ts"])
    assert "sponsor" in cols
    assert "unique_id" not in cols and "constant" not in cols


def test_nfa_never_lets_a_row_see_itself_or_its_present():
    import numpy as np
    from tabicl.scaling._relational import neighbour_aggregates
    # Causality by construction, not by check: an entity row's cutoff IS its timestamp and
    # the as-of scan counts strictly earlier rows, so the earliest row in every group must
    # see zero neighbours. No control can be fooled because there is nothing to fool.
    df = _nfa_frame()
    out = neighbour_aggregates(df, "ts", ["sponsor"], value_columns=["value"])
    cnt = next(c for c in out.columns if c.endswith("__count"))
    earliest = df.groupby("sponsor")["ts"].idxmin()
    assert (out.loc[earliest, cnt].to_numpy() == 0).all()


def test_nfa_aggregates_only_over_rows_sharing_the_value():
    import numpy as np
    import pandas as pd
    from tabicl.scaling._relational import neighbour_aggregates
    # Two groups with deliberately different values: each row's neighbour mean must come
    # from its OWN group. A join that crossed groups would land between 10 and 100.
    df = pd.DataFrame({
        "g": ["a", "a", "a", "b", "b", "b"],
        "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
        "v": [10.0, 10.0, 10.0, 100.0, 100.0, 100.0],
    })
    out = neighbour_aggregates(df, "ts", ["g"], value_columns=["v"])
    mean = next(c for c in out.columns if c.endswith("v__mean"))
    assert out[mean].iloc[2] == pytest.approx(10.0)     # third 'a' row sees two 'a' rows
    assert out[mean].iloc[5] == pytest.approx(100.0)    # third 'b' row sees two 'b' rows


def test_nfa_missing_values_do_not_become_a_category():
    import numpy as np
    import pandas as pd
    from tabicl.scaling._relational import neighbour_aggregates
    # Rows missing the grouping value share nothing. Treating NaN as a value would make
    # "missing" a category and join every incomplete row to every other -- a large, entirely
    # spurious neighbourhood.
    df = pd.DataFrame({
        "g": [None, None, None, "a", "a"],
        "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03",
                              "2020-01-01", "2020-01-02"]),
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    out = neighbour_aggregates(df, "ts", ["g"], value_columns=["v"])
    cnt = next(c for c in out.columns if c.endswith("__count"))
    assert (out[cnt].iloc[:3].to_numpy() == 0).all()    # the NaN rows see nobody
    assert out[cnt].iloc[4] == 1                        # the second 'a' row sees the first


def test_compression_fits_on_train_and_reuses_the_projection():
    import numpy as np
    from tabicl.scaling.eval_track_record import compress_fit_apply
    # Fitting per split would let validation and test each choose their own components, so
    # the arm validation SELECTS on would not be the arm test is judged on. That is the
    # exact shape of this project's val/test inversion, and it is why fit=False everywhere
    # but train.
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(size=(400, 40)), rng.normal(size=(120, 40))
    compress_fit_apply("arm", Xtr, 6, fit=True)
    a = compress_fit_apply("arm", Xva, 6, fit=False)
    b = compress_fit_apply("arm", Xva, 6, fit=False)
    assert a.shape == (120, 6)
    assert np.allclose(a, b)                       # deterministic, same projection twice


def test_compression_is_a_noop_when_the_arm_is_already_narrow():
    import numpy as np
    from tabicl.scaling.eval_track_record import compress_fit_apply
    # base at --max-columns 2 is already narrow; asking for more components than columns
    # must not error or invent dimensions.
    X = np.random.default_rng(0).normal(size=(50, 4))
    assert compress_fit_apply("narrow", X, 16, fit=True).shape == (50, 4)


def test_each_arm_gets_its_own_projection():
    import numpy as np
    from tabicl.scaling.eval_track_record import compress_fit_apply
    # Arms carry different columns. One shared projection would describe a feature set the
    # arm does not have.
    rng = np.random.default_rng(1)
    a_tr, b_tr = rng.normal(size=(200, 30)), rng.normal(size=(200, 50))
    compress_fit_apply("a", a_tr, 5, fit=True)
    compress_fit_apply("b", b_tr, 5, fit=True)
    ev = rng.normal(size=(20, 30))
    assert compress_fit_apply("a", ev, 5, fit=False).shape == (20, 5)


# --- context CONTENT: the axis never tested here ---------------------------------------
# Every context experiment in this project varied SIZE (--context-grid) or ORDER (random vs
# recent). Two independent measurements point at content instead: fixing the size axis cost
# no accuracy and halved the variance, and KernelICL (arXiv 2602.02162) measures a tabular
# ICL prediction as relying on only 11-29% of its context.

def test_knn_context_holds_length_fixed():
    import numpy as np
    from tabicl.scaling.eval_track_record import knn_context_indices
    # The comparison against a random draw must vary SELECTION and hold LENGTH fixed,
    # otherwise it is the context-size experiment again wearing a different name.
    rng = np.random.default_rng(0)
    X, Xe = rng.normal(size=(2000, 12)), rng.normal(size=(200, 12))
    idx = knn_context_indices(X, Xe, 300, seed=0)
    assert len(idx) == 300 and len(set(idx.tolist())) == 300


def test_knn_context_actually_retrieves_near_the_queries():
    import numpy as np
    from tabicl.scaling.eval_track_record import knn_context_indices
    # Queries are displaced from the pool centre; selected rows must sit closer to them
    # than the pool average, or the selection is not doing what it claims.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 10))
    Xe = rng.normal(size=(200, 10)) + 3.0
    idx = knn_context_indices(X, Xe, 400, seed=0)
    centre = Xe.mean(axis=0)
    assert (np.linalg.norm(X[idx] - centre, axis=1).mean()
            < np.linalg.norm(X - centre, axis=1).mean())


def test_knn_context_is_a_noop_when_the_pool_fits():
    import numpy as np
    from tabicl.scaling.eval_track_record import knn_context_indices
    rng = np.random.default_rng(0)
    X, Xe = rng.normal(size=(500, 8)), rng.normal(size=(50, 8))
    assert len(knn_context_indices(X, Xe, 5000, seed=0)) == 500


def test_knn_context_never_reads_labels():
    import inspect
    from tabicl.scaling.eval_track_record import knn_context_indices
    # Label-free by signature: there is no y parameter to leak through. The queries'
    # feature geometry is public at inference time; their labels are not.
    params = set(inspect.signature(knn_context_indices).parameters)
    assert not (params & {"y", "y_train", "y_test", "labels", "truth"})


def test_mmd_context_is_more_diverse_than_knn():
    import numpy as np
    from tabicl.scaling.eval_track_record import mmd_context_indices, knn_context_indices
    # The redundancy penalty is the ENTIRE difference from kNN, which is pure proximity.
    # Both should find the query mode; only MMD should spread out within it.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (2000, 8)), rng.normal(4, 1, (2000, 8))])
    Xe = rng.normal(4, 1, (300, 8))
    mmd = mmd_context_indices(X, Xe, 300, seed=0)
    knn = knn_context_indices(X, Xe, 300, seed=0)
    assert (mmd >= 2000).mean() > 0.9 and (knn >= 2000).mean() > 0.9   # both find the mode
    spread = lambda idx: np.linalg.norm(
        X[idx][:150, None] - X[idx][None, :150], axis=-1).mean()
    assert spread(mmd) > spread(knn)                                    # only MMD spreads


def test_mmd_context_holds_length_fixed_and_is_unique():
    import numpy as np
    from tabicl.scaling.eval_track_record import mmd_context_indices
    # No early stopping here. The paper's adaptive variant halts when MMD stops improving,
    # which saves inference cost and would silently turn this into the context-SIZE
    # experiment; the comparison must vary selection alone.
    rng = np.random.default_rng(0)
    X, Xe = rng.normal(size=(3000, 10)), rng.normal(size=(200, 10))
    idx = mmd_context_indices(X, Xe, 500, seed=0)
    assert len(idx) == 500 and len(set(idx.tolist())) == 500


def test_mmd_context_never_reads_labels():
    import inspect
    from tabicl.scaling.eval_track_record import mmd_context_indices
    params = set(inspect.signature(mmd_context_indices).parameters)
    assert not (params & {"y", "y_train", "y_test", "labels", "truth"})


# --- HyperTime lexicographic selection (arXiv 2305.18421) -------------------------------
# Keep configurations within a tolerance of the best AVERAGE validation score, then take the
# best WORST fold among them. Aimed at the measured regime: winning margin 0.35 against
# validation noise 0.69, with margin < noise in 35 of 41 blocks -- the argmax is usually
# choosing between candidates it cannot distinguish.

def test_lexi_breaks_ties_by_worst_fold_not_by_average():
    from tabicl.scaling.eval_track_record import lexi_select
    # +struct wins on average by 0.3, which is inside the noise floor; base has the better
    # worst fold. On rel-avito that is the correct direction -- validation prefers +struct
    # and test prefers base.
    cands = [(70.5, "+struct", 10000, "random"), (70.2, "base", 10000, "random")]
    sv = {("+struct", 10000, "random"): (71.0, 69.9),
          ("base", 10000, "random"): (70.4, 70.0)}
    pick, n, why = lexi_select(cands, sv, 0.01)
    assert pick[1] == "base" and n == 2
    assert "best worst fold" in why


def test_lexi_does_not_reach_outside_the_tolerance_band():
    from tabicl.scaling.eval_track_record import lexi_select
    # A configuration far below the best average must not win on worst-fold alone, or the
    # rule stops being a tiebreak and becomes a different objective.
    cands = [(70.5, "+struct", 10000, "random"), (60.0, "base", 10000, "random")]
    sv = {("+struct", 10000, "random"): (71.0, 69.9),
          ("base", 10000, "random"): (60.0, 60.0)}
    pick, n, _ = lexi_select(cands, sv, 0.01)
    assert pick[1] == "+struct" and n == 1


def test_lexi_falls_back_to_argmax_without_folds():
    from tabicl.scaling.eval_track_record import lexi_select
    # A task with no fold scores must behave exactly as before rather than erroring or
    # silently picking something arbitrary.
    cands = [(70.5, "+struct", 10000, "random"), (70.2, "base", 10000, "random")]
    pick, _, why = lexi_select(cands, {}, 0.01)
    assert pick[1] == "+struct" and "argmax" in why


def test_lexi_handles_no_candidates():
    from tabicl.scaling.eval_track_record import lexi_select
    pick, n, _ = lexi_select([], {}, 0.01)
    assert pick is None and n == 0


# --- the folds lexi reads have to actually be computed ----------------------------------
# `lexi_select` was tested and correct while the runner never fed it anything: the folds
# were gated on --abstain, so --select-lexi fell back to the plain argmax and reported a
# selection it had not made. The unit under test was never the one that was broken.


def test_select_lexi_alone_requests_the_folds():
    from tabicl.scaling.eval_track_record import fold_plan
    assert fold_plan(abstain=False, select_lexi=True,
                     cv_folds=0, gap_validation=False) is True


def test_abstain_alone_still_requests_the_folds():
    from tabicl.scaling.eval_track_record import fold_plan
    assert fold_plan(abstain=True, select_lexi=False,
                     cv_folds=0, gap_validation=False) is True


def test_neither_flag_leaves_the_folds_off():
    from tabicl.scaling.eval_track_record import fold_plan
    assert fold_plan(abstain=False, select_lexi=False,
                     cv_folds=0, gap_validation=False) is False


@pytest.mark.parametrize("cv_folds,gap", [(5, False), (0, True)])
def test_lexi_refuses_the_criteria_that_produce_no_folds(cv_folds, gap):
    from tabicl.scaling.eval_track_record import fold_plan
    with pytest.raises(SystemExit):
        fold_plan(abstain=False, select_lexi=True,
                  cv_folds=cv_folds, gap_validation=gap)


# --- the random validation slice must be redrawn every replicate ------------------------


def test_random_val_draw_varies_with_the_seed():
    """The defect this pins: one hardcoded draw shared by every replicate, so the reported
    spread covered context and model randomness but not the split itself."""
    from tabicl.scaling.eval_track_record import random_val_indices
    draws = [random_val_indices(9000, 1000, s) for s in range(5)]
    assert all(len(d) == 1000 for d in draws)
    assert len({d.tobytes() for d in draws}) == 5
    # and they must genuinely differ, not merely be unequal at one position
    assert len(set(draws[0].tolist()) & set(draws[1].tolist())) < 400


def test_random_val_draw_is_reproducible_for_a_given_seed():
    from tabicl.scaling.eval_track_record import random_val_indices
    a = random_val_indices(9000, 1000, 3)
    b = random_val_indices(9000, 1000, 3)
    assert np.array_equal(a, b)


def test_random_val_draw_never_takes_more_than_a_third_of_train():
    from tabicl.scaling.eval_track_record import random_val_indices
    idx = random_val_indices(900, 1000, 0)      # val is larger than train//3
    assert len(idx) == 300


def test_random_val_draw_is_sorted_unique_and_in_range():
    from tabicl.scaling.eval_track_record import random_val_indices
    idx = random_val_indices(5000, 400, 7)
    assert np.array_equal(idx, np.sort(idx))
    assert len(np.unique(idx)) == len(idx)
    assert idx.min() >= 0 and idx.max() < 5000


# --- the pod harness ---------------------------------------------------------------------
# Each of these pins a failure that cost a pod cycle. The harness used to live in a session
# scratchpad, so every one of them was rediscovered rather than remembered.


def test_payload_carries_the_runner_and_the_licence(tmp_path):
    """LICENSE is not optional -- pyproject declares `license = {file = "LICENSE"}`, so
    hatchling fails the build without it and the payload had never shipped it."""
    from tabicl.scaling import cycle
    out = tmp_path / "payload.tar.gz"
    cycle.build_payload(cycle.REPO, out)
    import tarfile
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    assert "LICENSE" in names
    assert "pyproject.toml" in names
    assert cycle.RUNNER in names
    cycle.verify_payload(cycle.REPO, out)          # must not raise


def test_payload_verification_catches_a_stale_archive(tmp_path):
    """The failure that cost four rounds was an archive that EXISTED and was stale, so the
    check has to compare content, not presence."""
    from tabicl.scaling import cycle
    out = tmp_path / "payload.tar.gz"
    fake = tmp_path / "repo"
    (fake / "src/tabicl/scaling").mkdir(parents=True)
    (fake / cycle.RUNNER).write_bytes(b"# current\n")
    cycle.build_payload(fake, out, parts=("src",))
    cycle.verify_payload(fake, out)                # matches
    (fake / cycle.RUNNER).write_bytes(b"# edited after the archive was built\n")
    with pytest.raises(SystemExit):
        cycle.verify_payload(fake, out)


def test_payload_verification_catches_a_missing_runner(tmp_path):
    from tabicl.scaling import cycle
    out = tmp_path / "payload.tar.gz"
    fake = tmp_path / "repo"
    (fake / "src").mkdir(parents=True)
    (fake / "src/unrelated.py").write_bytes(b"x\n")
    cycle.build_payload(fake, out, parts=("src",))
    with pytest.raises(SystemExit):
        cycle.verify_payload(fake, out)


def test_shell_scripts_are_normalised_to_lf():
    """A CRLF `_pod_setup.sh` made bash read `set -euo pipefail\\r` and report
    "pipefail: invalid option name", which reads like the wrong shell rather than the wrong
    file."""
    from tabicl.scaling import cycle
    assert cycle.lf_bytes(b"set -e\r\nexit 0\r\n") == b"set -e\nexit 0\n"
    assert cycle.lf_bytes(b"already\nlf\n") == b"already\nlf\n"


@pytest.mark.parametrize("name", ["_pod_setup.sh", "_pod_remote.sh"])
def test_shipped_scripts_have_no_carriage_returns_in_the_repo(name):
    import pathlib
    from tabicl.scaling import cycle
    p = pathlib.Path(cycle.__file__).with_name(name)
    assert b"\r" not in p.read_bytes(), f"{name} is CRLF and will fail on the host"


def test_setup_does_not_truncate_pip_output():
    """`pip ... | tail -3` delivered "note: This is an issue with the package mentioned
    above" with the package, and the reason, cut off. Output filters have eaten results on
    this project five times."""
    import pathlib
    from tabicl.scaling import cycle
    setup = pathlib.Path(cycle.__file__).with_name("_pod_setup.sh").read_text()
    for line in setup.splitlines():
        if line.strip().startswith("pip install"):
            assert "tail" not in line, f"pip output is truncated: {line!r}"


# --- temporal-distance extrapolation -----------------------------------------------------


def test_extrapolation_prefers_the_candidate_that_holds_up_with_distance():
    """The whole point: a candidate that scores best nearby and decays, against one that is
    slightly worse nearby and flat. Validation only ever sees the nearby end."""
    from tabicl.scaling.eval_track_record import extrapolate_select
    decays = [(4.0, 70.0), (8.0, 68.0), (12.0, 66.0)]      # -0.5/day
    flat = [(4.0, 69.0), (8.0, 69.0), (12.0, 69.0)]        # 0.0/day
    by_gap = {"decays": decays, "flat": flat}
    # Ordinary validation, sitting at the near end, would take "decays" (70.0 vs 69.0).
    assert max(by_gap, key=lambda k: by_gap[k][0][1]) == "decays"
    pick, reason, diag = extrapolate_select(by_gap, gap_test=24.0, shrink_slope=False)
    assert pick == "flat"
    assert diag["lever"] > 0          # test sits beyond the measured range


def test_extrapolation_says_when_the_target_sits_at_the_centre_of_its_bracket():
    """The default 0.5x/1.0x/1.5x bracket puts the target exactly at the centroid, and a
    least-squares line passes through its centroid -- so the slope term is zero identically,
    whatever the slope. That is correct behaviour (measuring AT the target beats extrapolating
    to it) but it must not be reported as a shrinkage failure."""
    from tabicl.scaling.eval_track_record import extrapolate_select
    steep = {"a": [(10.0, 60.0), (20.0, 70.0), (30.0, 80.0)]}     # a very large slope
    pick, reason, diag = extrapolate_select(steep, gap_test=20.0)
    assert diag["adjustments"]["a"] == pytest.approx(0.0)
    assert "GAP-MATCHED" in reason and "zero by construction" in reason
    assert "shrank" not in reason


def test_extrapolation_shrinks_a_slope_its_own_points_cannot_support():
    """A slope fitted through noise must not be trusted at a long lever arm; this project's
    screened estimates regress hard (+0.66 -> +0.39, +0.60 -> +0.09)."""
    from tabicl.scaling.eval_track_record import extrapolate_select
    noisy = {"a": [(4.0, 70.0), (8.0, 62.0), (12.0, 74.0)]}     # wild residuals
    _, _, diag = extrapolate_select(noisy, gap_test=100.0, shrink_slope=True)
    shrunk = abs(diag["adjustments"]["a"])
    _, _, diag_raw = extrapolate_select(noisy, gap_test=100.0, shrink_slope=False)
    assert shrunk < abs(diag_raw["adjustments"]["a"]) / 10


def test_extrapolation_discards_a_slope_from_only_two_points():
    """Two points fit a line exactly and leave no residual, so nothing about the slope is
    verifiable. Trusting it *because* its error is unmeasurable is precisely backwards."""
    from tabicl.scaling.eval_track_record import extrapolate_select
    two = {"steep": [(2.0, 60.0), (5.0, 75.0)], "flat": [(2.0, 66.0), (5.0, 66.0)]}
    pick, reason, diag = extrapolate_select(two, gap_test=10.0)
    # A 5-points-per-day slope would swamp everything if it were trusted; it is discarded.
    assert diag["adjustments"]["steep"] == 0.0
    assert diag["max_abs_adjustment"] == 0.0
    # With every adjustment zero the ranking is by mean alone: steep 67.5 beats flat 66.0.
    assert pick == "steep"
    assert "not an extrapolation" in reason


def test_extrapolation_degenerates_to_mean_ranking_when_it_must_say_so():
    from tabicl.scaling.eval_track_record import extrapolate_select
    one_point = {"a": [(4.0, 70.0)], "b": [(4.0, 71.0)]}
    pick, reason, _ = extrapolate_select(one_point, gap_test=24.0)
    assert pick == "b"
    assert "not an extrapolation" in reason


def test_extrapolation_reports_how_far_past_its_data_it_is_reading():
    """`lever` is the honest health warning: 2.0 means the line is trusted twice as far as it
    was fitted, which is where a linear extrapolation starts inventing numbers."""
    from tabicl.scaling.eval_track_record import extrapolate_select
    by_gap = {"a": [(10.0, 70.0), (20.0, 69.0)]}
    _, _, diag = extrapolate_select(by_gap, gap_test=40.0)
    assert diag["span"] == 10.0
    assert diag["lever"] == pytest.approx(2.0)


def test_extrapolation_pools_hold_their_size_constant():
    """The confound that killed --gap-validation: it carved pools by a time cutoff and let
    the size fall out, selecting on 11,522 rows against a 19,239-row final fit. A slope
    measured that way is partly a slope in pool size, which is not a quantity that differs
    between validation and test."""
    from tabicl.scaling.eval_track_record import extrapolation_pools
    t = np.arange(1000, dtype=float)                      # one row per unit time
    built, unusable = extrapolation_pools(t, gaps=[10, 50, 100], pool_size=200, query_size=100)
    assert len(built) == 3 and not unusable
    assert {len(p) for _, p, _ in built} == {200}


def test_extrapolation_pools_score_the_same_query_rows_at_every_gap():
    """Every gap must score the SAME rows, or a difference between gaps could be a difference
    between query populations."""
    from tabicl.scaling.eval_track_record import extrapolation_pools
    t = np.arange(1000, dtype=float)
    built, _ = extrapolation_pools(t, gaps=[10, 50, 100], pool_size=200, query_size=100)
    queries = [tuple(sorted(q.tolist())) for _, _, q in built]
    assert len(set(queries)) == 1


def test_extrapolation_pools_grow_staler_and_never_touch_the_query():
    from tabicl.scaling.eval_track_record import extrapolation_pools
    t = np.arange(1000, dtype=float)
    built, _ = extrapolation_pools(t, gaps=[10, 50, 100], pool_size=200, query_size=100)
    q = set(built[0][2].tolist())
    last_end = None
    for g, pool, _ in built:
        assert not (set(pool.tolist()) & q)              # a row is never fitted and queried
        end = t[pool].max()
        if last_end is not None:
            assert end < last_end                        # bigger gap => staler pool
        last_end = end


def test_extrapolation_pool_size_is_set_by_the_scarcest_gap_not_the_dataset():
    """Sizing the pool at min(--context, n_train) looks reasonable and is wrong: the query
    slice consumes rows and staler gaps consume more, so the binding constraint is the
    scarcest gap. On rel-trial that mistake asked for 10,000-row pools where the widest gap
    could offer 6,483 -- every gap unbuildable, at any staleness."""
    from tabicl.scaling.eval_track_record import extrapolation_pool_size, extrapolation_pools
    t = np.arange(12000, dtype=float)
    size, avail = extrapolation_pool_size(t, gaps=[366, 731, 1096], query_size=2000, cap=10000)
    assert dict(avail)[1096.0] == size          # the scarcest gap sets it
    assert size < 10000
    # and at that size every gap is buildable, which is the whole point
    built, unusable = extrapolation_pools(t, [366, 731, 1096], size, 2000)
    assert len(built) == 3 and not unusable


def test_extrapolation_pool_size_respects_the_context_cap():
    from tabicl.scaling.eval_track_record import extrapolation_pool_size
    t = np.arange(500000, dtype=float)          # plenty of rows; the cap should bind instead
    size, _ = extrapolation_pool_size(t, gaps=[10, 20], query_size=2000, cap=10000)
    assert size == 10000


def test_extrapolation_pools_drop_gaps_that_select_an_identical_pool():
    """rel-avito's timestamps are coarse enough that its 2-day and 3.5-day cutoffs fall
    between the same pair of dates and yield the SAME pool. Fitting the slope through that
    duplicated point shrinks the residual, understates the slope's standard error, and
    disarms the shrinkage precisely where it is needed."""
    from tabicl.scaling.eval_track_record import extrapolation_pools
    # daily granularity: every row shares one of 40 timestamps
    t = (np.arange(4000, dtype=float) // 100)        # 40 distinct timestamps, 100 rows each
    # 2.2 and 2.4 both land between the same pair of timestamps, so both cutoffs admit
    # exactly the rows at t <= 35 -- one measurement wearing two labels.
    built, unusable = extrapolation_pools(t, gaps=[2.2, 2.4, 8.0], pool_size=500, query_size=200)
    gaps_kept = [g for g, _, _ in built]
    assert 2.2 in gaps_kept and 8.0 in gaps_kept
    assert 2.4 not in gaps_kept                      # same pool as the 2.2 gap
    assert (2.4, -1) in unusable                     # reported as a duplicate, not a build failure
    pools = [p.tobytes() for _, p, _ in built]
    assert len(set(pools)) == len(pools)             # no two retained gaps share a pool


def test_extrapolation_gap_is_applied_at_hour_resolution():
    """`int(3.5)` truncated a 3.5-day request to 3 days while the regression kept using 3.5
    as the abscissa -- a slope wrong by exactly that ratio."""
    from tabicl.scaling.eval_track_record import extrapolation_pools
    t0 = np.datetime64("2024-01-01T00:00:00")
    t = t0 + np.arange(4000) * np.timedelta64(1, "h")       # hourly, 166 days
    built, _ = extrapolation_pools(t, gaps=[3.0, 3.5], pool_size=200, query_size=200)
    assert [g for g, _, _ in built] == [3.0, 3.5]           # a half day is a real difference
    ends = [t[p].max() for _, p, _ in built]
    assert (ends[0] - ends[1]) == np.timedelta64(12, "h")   # exactly half a day apart


def test_extrapolation_pools_report_gaps_they_could_not_build():
    """Dropping a gap silently would shorten the lever arm without saying so -- the same
    'not applicable' vs 'measured at no effect' confusion this project keeps hitting."""
    from tabicl.scaling.eval_track_record import extrapolation_pools
    t = np.arange(300, dtype=float)
    built, unusable = extrapolation_pools(t, gaps=[10, 250], pool_size=100, query_size=50)
    assert [g for g, _, _ in built] == [10.0]
    assert unusable and unusable[0][0] == 250.0


def test_extrapolation_handles_no_candidates():
    from tabicl.scaling.eval_track_record import extrapolate_select
    pick, reason, diag = extrapolate_select({}, gap_test=10.0)
    assert pick is None and diag == {}


def test_extra_pip_is_followed_by_a_numpy_torch_bridge_check():
    """Drift-Resilient TabPFN installed cleanly and pulled numpy 2.x under a torch compiled
    against 1.x. Torch then imported with "Failed to initialize NumPy: _ARRAY_API not found"
    -- a warning, not an exception -- so the import gate passed and the round started with
    every tensor<->array conversion dead. An import that succeeds is not a working env."""
    import pathlib
    from tabicl.scaling import cycle
    setup = pathlib.Path(cycle.__file__).with_name("_pod_setup.sh").read_text()
    extra = setup.split("TABICL_EXTRA_PIP", 1)[1]
    assert "from_numpy" in extra and ".numpy()" in extra, \
        "an extra install must be followed by a check that torch and numpy still interoperate"


def test_poller_does_not_read_a_dead_host_as_a_finished_round():
    """A community host was reclaimed nine minutes into a five-hour round. ssh printed
    "Connection refused", that text was partitioned as if it were a probe reply, and the
    poller announced WORK_DONE and tore the round down."""
    from tabicl.scaling.cycle import parse_probe
    assert parse_probe(255, "ssh: connect to host 1.2.3.4 port 22: Connection refused") is None
    assert parse_probe(0, "") is None
    assert parse_probe(0, "WORK_DONE 2026-08-10") is None          # no separators: not a reply


def test_probe_process_count_cannot_match_itself():
    """`pgrep -f eval_track_record` matched the probe's own ssh command line, so the count was
    never below one on a healthy host and the "nothing running" break could never fire. A lane
    whose work had been stopped polled to its deadline, billing the whole way."""
    from tabicl.scaling.cycle import PROBE
    assert "pgrep -fc '[e]val_track_rec'" in PROBE
    assert "pgrep -f eval_track_record" not in PROBE


def test_poller_parses_a_real_probe_reply():
    from tabicl.scaling.cycle import parse_probe
    out = ("WORK_DONE 2026-08-10T22:00:00Z\n---\nclicks_base rc=0\nclicks_rvs rc=0\n"
           "---\n0\n---\n############ clicks_rvs rc=0 ############\n")
    done, finished, running, tail = parse_probe(0, out)
    assert "WORK_DONE" in done
    assert finished == ["clicks_base", "rc=0", "clicks_rvs", "rc=0"]
    assert running == "0"
    assert "clicks_rvs" in tail


def test_poller_reports_a_round_still_running():
    from tabicl.scaling.cycle import parse_probe
    done, finished, running, _ = parse_probe(0, "\n---\n\n---\n2\n---\nbase  val=63.46\n")
    assert done == "" and finished == [] and running == "2"


def test_supported_kwargs_drops_what_a_constructor_does_not_take():
    """Discovering a class and then guessing its constructor is the same mistake one level
    down; it cost a pod cycle on TabFM."""
    from tabicl.scaling.eval_backbone import supported_kwargs

    class TakesNeither:
        def __init__(self, n_estimators=1): ...

    class TakesDevice:
        def __init__(self, device=None, n_estimators=1): ...

    class TakesAnything:
        def __init__(self, **kw): ...

    assert supported_kwargs(TakesNeither, device="cuda:0", random_state=3) == {}
    assert supported_kwargs(TakesDevice, device="cuda:0", random_state=3) == {"device": "cuda:0"}
    # **kwargs means we cannot tell -- pass through rather than silently dropping the device
    assert supported_kwargs(TakesAnything, device="cuda:0") == {"device": "cuda:0"}


def test_setup_gate_runs_the_entry_point_not_a_version_banner():
    """A version banner aborted setup under `set -e`; diagnostics may not fail a build, so
    the real gate is the entry point the round actually calls."""
    import pathlib
    from tabicl.scaling import cycle
    setup = pathlib.Path(cycle.__file__).with_name("_pod_setup.sh").read_text()
    assert "eval_track_record --help" in setup
    # SETUP_OK must be the LAST thing emitted: the gate on the host greps for it, so a line
    # printed before the real check would announce a setup that had not happened yet.
    last = [ln for ln in setup.splitlines() if ln.strip()][-1]
    assert "SETUP_OK" in last, f"SETUP_OK is not the final line; got {last!r}"
