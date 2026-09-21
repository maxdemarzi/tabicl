"""Tests for TP-04: QK-norm (RMSNorm on queries and keys).

The TabPFN-3.5 report adds this to keep training stable at the doubled model width and
across joint classification/regression training, so it is a prerequisite for TP-06 and TP-07
rather than a standalone win. None of that is checkable without training.

What IS checkable, and what these tests are for, is that it does not break the KV cache. The
normalization has to happen on the same side of the cache boundary for a stored key as for a
freshly computed one: `k` is normed and RoPE'd before being returned for caching, so the
cached path must normalize `q` only. Get that wrong and cached and uncached predictions
diverge silently -- on a feature this project advertises.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabicl._model.attention import RMSNorm
from tabicl._model.layers import MultiheadAttention
from tabicl._model.tabicl import TabICL


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


def test_rmsnorm_normalizes_the_last_dimension():
    norm = RMSNorm(8)
    x = torch.randn(3, 5, 8) * 17.0
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)


def test_rmsnorm_is_scale_invariant():
    """The point of the thing: it removes magnitude, which is what destabilizes logits."""

    norm = RMSNorm(8)
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(norm(x), norm(x * 100.0), rtol=1e-3, atol=1e-3)


def test_rmsnorm_preserves_dtype():
    norm = RMSNorm(8)
    assert norm(torch.randn(2, 8, dtype=torch.float16)).dtype == torch.float16


def test_rmsnorm_weight_is_learnable():
    norm = RMSNorm(8)
    norm(torch.randn(2, 8)).sum().backward()
    assert norm.weight.grad is not None


# --------------------------------------------------------------------------- #
# attention wiring
# --------------------------------------------------------------------------- #


def test_attention_holds_no_norm_by_default():
    assert MultiheadAttention(64, 4).qk_norm is None


def test_attention_norm_is_over_head_dim_not_model_dim():
    """Per-head vectors are what get normalized -- embed_dim // num_heads."""

    attn = MultiheadAttention(64, 4, qk_norm=True)
    assert attn.qk_norm.weight.shape == (16,)


def test_qk_norm_changes_the_output():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 64)
    off = MultiheadAttention(64, 4, qk_norm=False)
    on = MultiheadAttention(64, 4, qk_norm=True)
    on.load_state_dict(off.state_dict(), strict=False)
    assert not torch.allclose(off(x, x, x), on(x, x, x), atol=1e-5)


# --------------------------------------------------------------------------- #
# the cache boundary -- the reason these tests exist
# --------------------------------------------------------------------------- #


def _cached_vs_uncached(attn, query, key, value):
    """Run attention normally, then again reusing the returned K/V as a cache."""

    from tabicl._model.kv_cache import KVCacheEntry

    plain, k, v = attn(query, key, value, need_kv=True)
    cached = attn(query, cached_kv=KVCacheEntry(key=k, value=v))
    return plain, cached


@pytest.mark.parametrize("qk_norm", [False, True])
def test_cached_and_uncached_attention_agree(qk_norm):
    """Must hold both ways: TP-04 is only safe if it does not move this boundary."""

    torch.manual_seed(0)
    attn = MultiheadAttention(64, 4, qk_norm=qk_norm).eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        plain, cached = _cached_vs_uncached(attn, x, x, x)
    torch.testing.assert_close(plain, cached, rtol=1e-5, atol=1e-5)


def test_qk_norm_is_not_applied_twice_to_cached_keys():
    """A re-normed key would still be finite and plausible -- just wrong.

    Applying RMSNorm twice is not idempotent once the learned weight is not all ones, so
    setting a distinctive weight makes double application detectable.
    """

    torch.manual_seed(0)
    attn = MultiheadAttention(64, 4, qk_norm=True).eval()
    with torch.no_grad():
        attn.qk_norm.weight.copy_(torch.linspace(0.5, 2.0, 16))

    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        plain, cached = _cached_vs_uncached(attn, x, x, x)
    torch.testing.assert_close(plain, cached, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# model wiring
# --------------------------------------------------------------------------- #


def test_flag_is_off_by_default():
    assert sum(1 for m in TabICL(max_classes=10).modules() if isinstance(m, RMSNorm)) == 0


def test_flag_reaches_every_attention_block():
    """3 column blocks x 2 (ISAB is two-stage) + 3 row + 12 ICL = 21."""

    model = TabICL(max_classes=10, qk_norm=True)
    assert sum(1 for m in model.modules() if isinstance(m, RMSNorm)) == 21


def test_parameter_cost_is_negligible():
    off = sum(p.numel() for p in TabICL(max_classes=10).parameters())
    on = sum(p.numel() for p in TabICL(max_classes=10, qk_norm=True).parameters())
    assert 0 < (on - off) < 0.001 * off


def test_forward_pass_runs_and_is_finite():
    torch.manual_seed(0)
    model = TabICL(max_classes=10, qk_norm=True).eval()
    X = torch.randn(1, 40, 6)
    y_train = torch.randint(0, 3, (1, 30)).float()
    with torch.no_grad():
        out = model(X, y_train, d=torch.tensor([6]))
    assert torch.isfinite(out).all()


def test_composes_with_fourier_value_encoding():
    """TP-01 and TP-04 land in the same Phase 2/3 sequence and must not conflict."""

    torch.manual_seed(0)
    model = TabICL(max_classes=10, qk_norm=True, col_fourier_value=True, col_fourier_freqs=8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 40, 6), torch.randint(0, 3, (1, 30)).float(), d=torch.tensor([6]))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# input_norm: the report's "LayerNorm after the input encoding"
# --------------------------------------------------------------------------- #


def test_input_norm_off_adds_nothing():
    """Off must leave the state dict exactly as it was, or released checkpoints stop loading."""

    assert TabICL(max_classes=10).state_dict().keys() == TabICL(max_classes=10, input_norm=False).state_dict().keys()
    assert not any("in_norm" in k for k in TabICL(max_classes=10).state_dict())


def test_input_norm_normalizes_the_cell_encoding():
    torch.manual_seed(0)
    emb = TabICL(max_classes=10, input_norm=True).col_embedder
    assert isinstance(emb.in_norm, torch.nn.LayerNorm)
    with torch.no_grad():
        out = emb._encode_cells(torch.randn(3, 20, emb.feature_group_size) * 50.0)  # grouped cells
    torch.testing.assert_close(out.mean(-1), torch.zeros(3, 20), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(-1, unbiased=False), torch.ones(3, 20), atol=1e-3, rtol=0)


def test_input_norm_follows_bias_free_ln():
    """The regression recipe's bias-free LayerNorm must reach the new norm too."""

    emb = TabICL(max_classes=10, input_norm=True, bias_free_ln=True).col_embedder
    assert emb.in_norm.bias is None


@pytest.mark.parametrize("mode", ["kv", "repr"])
def test_input_norm_cached_and_uncached_agree(mode):
    """Per-token, so it cannot straddle the cache boundary -- asserted, not assumed."""

    torch.manual_seed(0)
    model = TabICL(max_classes=10, input_norm=True, zero_init=False).eval()
    with torch.no_grad():
        model.col_embedder.in_norm.weight.copy_(torch.linspace(0.5, 2.0, model.embed_dim))
    X = torch.randn(1, 40, 6)
    y = torch.arange(32).remainder(3).float().unsqueeze(0)
    with torch.no_grad():
        plain = model(X, y)
        stored = model.forward_with_cache(X_train=X[:, :32], y_train=y, X_test=X[:, 32:],
                                          store_cache=True, cache_mode=mode)
        reused = model.forward_with_cache(X_test=X[:, 32:], use_cache=True, store_cache=False)
    torch.testing.assert_close(stored, plain, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reused, plain, rtol=1e-5, atol=1e-5)


def test_input_norm_composes_with_multitask_equivalence():
    """A joint model loaded from a single-task one stays identical with the norm on."""

    torch.manual_seed(0)
    single = TabICL(max_classes=0, input_norm=True, zero_init=False).eval()
    joint = TabICL(max_classes=10, multitask=True, input_norm=True, zero_init=False)
    joint.load_single_task_state_dict(single.state_dict(), "regression")
    joint.set_task("regression").eval()
    X, y = torch.randn(1, 40, 6), torch.randn(1, 32)
    with torch.no_grad():
        torch.testing.assert_close(joint(X, y), single(X, y), rtol=1e-5, atol=1e-5)
