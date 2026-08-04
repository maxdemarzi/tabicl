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

    report = temporal_control(lambda shift: 0.80 - 0.03 * shift, shifts=(0.0, 30.0))
    assert report.passed
    assert report.observed == pytest.approx(0.80)


def test_temporal_control_requires_an_unshifted_baseline():
    from tabicl.scaling import temporal_control

    with pytest.raises(ValueError, match="must start at 0.0"):
        temporal_control(lambda s: 0.5, shifts=(30.0, 90.0))
