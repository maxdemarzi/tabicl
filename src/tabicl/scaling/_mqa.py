"""Multi-query attention for the ICL KV cache.

TabPFN-3 (arXiv 2605.13986, Section 2.4.2) shrinks its KV cache by having test-row
queries attend to train-row keys/values through a *single* KV head, while train rows
keep full multi-head attention. The reduction is exactly ``nhead``.

Scoping caveat
--------------
TabPFN-3 was *pretrained* this way. TabICLv2's released checkpoint has 8 full KV
heads, so collapsing to one head post hoc is an approximation, not a refactor.
A faithful port needs pretraining with ``icl_nhead_kv=1``.

What this module provides today:

* ``kv_cache_bytes`` -- measure the cache, so the win is quantified rather than
  assumed;
* ``collapse_kv_heads`` -- post-hoc head collapse ('mean' or 'first') producing a
  real 8x smaller cache, whose accuracy cost can then be measured honestly.
"""

from __future__ import annotations

from typing import Literal

import torch

from .._model.kv_cache import KVCache, KVCacheEntry

__all__ = ["kv_cache_bytes", "collapse_kv_heads", "expand_kv_heads"]


def kv_cache_bytes(cache: KVCache) -> int:
    """Total bytes held by a KV cache's key/value tensors."""
    total = 0
    for entry in cache.kv.values():
        for t in (entry.key, entry.value):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def collapse_kv_heads(cache: KVCache, mode: Literal["mean", "first"] = "mean") -> KVCache:
    """Collapse an ``nhead``-head KV cache to a single KV head (multi-query).

    Parameters
    ----------
    cache : KVCache
        Cache whose entries hold ``(..., nhead, src_len, head_dim)`` tensors.

    mode : {'mean', 'first'}, default='mean'
        How to reduce the head axis. ``'mean'`` averages the heads;
        ``'first'`` keeps head 0. Neither is equivalent to training with MQA --
        this trades accuracy for an ``nhead``x smaller cache.

    Returns
    -------
    KVCache
        New cache with a single KV head per entry.
    """
    if mode not in ("mean", "first"):
        raise ValueError(f"mode must be 'mean' or 'first', got {mode!r}")

    out = KVCache()
    for layer_idx, entry in cache.kv.items():
        if entry.key is None:
            out.kv[layer_idx] = entry
            continue
        if mode == "mean":
            k = entry.key.mean(dim=-3, keepdim=True)
            v = entry.value.mean(dim=-3, keepdim=True)
        else:
            k = entry.key[..., :1, :, :].clone()
            v = entry.value[..., :1, :, :].clone()
        out.kv[layer_idx] = KVCacheEntry(key=k, value=v)
    return out


def expand_kv_heads(cache: KVCache, nhead: int) -> KVCache:
    """Broadcast a single-KV-head cache back to ``nhead`` heads.

    Scaled dot-product attention needs query and key head counts to agree. Real MQA
    kernels broadcast without materialising; this expand keeps the *stored* cache
    small (the actual saving) while staying compatible with the stock attention
    path, at the cost of a transient view-expand during the forward pass.
    """
    out = KVCache()
    for layer_idx, entry in cache.kv.items():
        if entry.key is None:
            out.kv[layer_idx] = entry
            continue
        k = entry.key.expand(*entry.key.shape[:-3], nhead, *entry.key.shape[-2:])
        v = entry.value.expand(*entry.value.shape[:-3], nhead, *entry.value.shape[-2:])
        out.kv[layer_idx] = KVCacheEntry(key=k, value=v)
    return out
