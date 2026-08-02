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
