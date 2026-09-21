"""Tests for TP-05: grouped-query attention in the in-context transformer.

The KV cache is what bounds cached inference -- measured at 24 KiB per training row per
estimator in fp16 (benchmarks/RESULTS.md), i.e. 12 ICL blocks x 2 x 512 dim. Projecting keys
and values to fewer heads shrinks exactly that, and they are expanded back to full width for
the attention itself, so the computation is unchanged and only the stored tensors shrink.

The load-bearing property is that cached and uncached predictions still agree: the cache is a
headline feature, and a mismatch here would be silent.
"""

from __future__ import annotations

import pytest
import torch

from tabicl._model.kv_cache import KVCacheEntry
from tabicl._model.layers import MultiheadAttention
from tabicl._model.tabicl import TabICL


def _kv_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


# --------------------------------------------------------------------------- #
# attention module
# --------------------------------------------------------------------------- #


def test_off_by_default():
    attn = MultiheadAttention(64, 8)
    assert attn.num_kv_heads is None
    assert attn.q_proj is None and attn.kv_proj is None
    assert attn.in_proj_weight is not None


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_cache_shrinks_in_proportion(num_kv_heads):
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        _, k_full, v_full = MultiheadAttention(64, 8).eval()(x, x, x, need_kv=True)
        _, k_gqa, v_gqa = MultiheadAttention(64, 8, num_kv_heads=num_kv_heads).eval()(
            x, x, x, need_kv=True)
    assert _kv_bytes(k_full, v_full) / _kv_bytes(k_gqa, v_gqa) == pytest.approx(8 / num_kv_heads)


@pytest.mark.parametrize("num_kv_heads", [None, 1, 2, 8])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_cached_and_uncached_agree(num_kv_heads, qk_norm):
    """Must hold for every combination -- TP-04 and TP-05 both touch the cache boundary."""

    torch.manual_seed(0)
    attn = MultiheadAttention(64, 8, qk_norm=qk_norm, num_kv_heads=num_kv_heads).eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        plain, k, v = attn(x, x, x, need_kv=True)
        cached = attn(x, cached_kv=KVCacheEntry(key=k, value=v))
    torch.testing.assert_close(plain, cached, rtol=1e-5, atol=1e-5)


def test_output_shape_is_unchanged():
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        assert MultiheadAttention(64, 8, num_kv_heads=1).eval()(x, x, x).shape == x.shape


def test_num_kv_heads_must_divide_num_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiheadAttention(64, 8, num_kv_heads=3)


def test_equal_heads_is_the_standard_path():
    """num_kv_heads == num_heads must not switch to the separate projections."""

    attn = MultiheadAttention(64, 8, num_kv_heads=8)
    assert attn.q_proj is None and attn.in_proj_weight is not None


def test_the_unused_packed_projection_is_dropped():
    """Left in place it is never read and still costs 3*embed_dim^2 per block."""

    attn = MultiheadAttention(64, 8, num_kv_heads=1)
    assert attn.in_proj_weight is None
    assert "in_proj_weight" not in dict(attn.named_parameters())


def test_gradients_flow_through_both_projections():
    attn = MultiheadAttention(64, 8, num_kv_heads=2)
    attn(torch.randn(1, 6, 64), torch.randn(1, 6, 64), torch.randn(1, 6, 64)).sum().backward()
    assert attn.q_proj.weight.grad is not None
    assert attn.kv_proj.weight.grad is not None


# --------------------------------------------------------------------------- #
# model wiring
# --------------------------------------------------------------------------- #


def test_model_flag_is_off_by_default():
    model = TabICL(max_classes=10)
    assert model.icl_predictor.tf_icl.blocks[0].attn.num_kv_heads is None


def test_model_flag_reaches_every_icl_block():
    model = TabICL(max_classes=10, icl_num_kv_heads=1)
    blocks = model.icl_predictor.tf_icl.blocks
    assert all(b.attn.num_kv_heads == 1 for b in blocks)


def test_parameter_count_falls():
    """GQA removes more from the packed projection than it adds back."""

    off = sum(p.numel() for p in TabICL(max_classes=10).parameters())
    on = sum(p.numel() for p in TabICL(max_classes=10, icl_num_kv_heads=1).parameters())
    assert on < off * 0.85


def test_forward_and_cached_predict_run():
    torch.manual_seed(0)
    model = TabICL(max_classes=10, icl_num_kv_heads=1).eval()
    X_train, y_train = torch.randn(1, 64, 6), torch.randint(0, 3, (1, 64)).float()
    X_test = torch.randn(1, 4, 6)
    with torch.no_grad():
        model.forward_with_cache(X_train=X_train, y_train=y_train, X_test=X_test,
                                 store_cache=True, use_cache=False)
        out = model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False)
    assert torch.isfinite(out).all()


def test_composes_with_qk_norm_and_fourier():
    torch.manual_seed(0)
    model = TabICL(max_classes=10, icl_num_kv_heads=1, qk_norm=True,
                   col_fourier_value=True, col_fourier_freqs=8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 40, 6), torch.randint(0, 3, (1, 30)).float(), d=torch.tensor([6]))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# TP-06 interaction: why GQA is the prerequisite for widening
# --------------------------------------------------------------------------- #


def _icl_cache_bytes(model, n_train=256, n_feat=8):
    X_train = torch.randn(1, n_train, n_feat)
    y_train = torch.randint(0, 3, (1, n_train)).float()
    X_test = torch.randn(1, 4, n_feat)
    with torch.no_grad():
        model.forward_with_cache(X_train=X_train, y_train=y_train, X_test=X_test,
                                 store_cache=True, use_cache=False)
    sub = model._cache.icl_cache
    return sum(t.numel() * t.element_size()
               for e in sub.kv.values() for t in (e.key, e.value) if t is not None)


def test_without_gqa_doubling_width_doubles_the_cache():
    """The problem TP-06 creates on its own."""

    narrow = _icl_cache_bytes(TabICL(max_classes=10, row_num_cls=4, icl_nhead=8).eval())
    wide = _icl_cache_bytes(TabICL(max_classes=10, row_num_cls=8, icl_nhead=16).eval())
    assert wide == pytest.approx(2 * narrow, rel=0.01)


def test_with_gqa_the_cache_stops_depending_on_width():
    """The reason TP-05 gates TP-06.

    With one KV head the cache is blocks x 2 x head_dim, and head_dim is held at 64 as width
    grows, so widening becomes free for the cache instead of doubling it.
    """

    narrow = _icl_cache_bytes(
        TabICL(max_classes=10, row_num_cls=4, icl_nhead=8, icl_num_kv_heads=1).eval())
    wide = _icl_cache_bytes(
        TabICL(max_classes=10, row_num_cls=8, icl_nhead=16, icl_num_kv_heads=1).eval())
    assert wide == pytest.approx(narrow, rel=0.01)


def test_the_widened_model_actually_trains():
    """Constructible is not the same as trainable."""

    torch.manual_seed(0)
    model = TabICL(max_classes=10, row_num_cls=8, icl_nhead=16, icl_num_kv_heads=1)
    model.train()
    X = torch.randn(1, 96, 8)
    y_train = torch.randint(0, 3, (1, 64)).float()
    out = model(X, y_train, d=torch.tensor([8]))
    out.float().square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
