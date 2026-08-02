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
