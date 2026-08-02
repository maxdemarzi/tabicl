"""Row-chunked column embedding.

Ports the two-phase inference scheme of TabPFN-3 (arXiv 2605.13986, Section 2.4.1)
onto TabICLv2's ``ColEmbedding``. The pre-ICL stage materialises an
``(N, C, d)`` activation, so peak memory grows with rows x columns and saturates
the GPU long before anything becomes compute-bound. TabICLv2's existing answer is
to offload activations to CPU or disk; this streams the row axis instead and keeps
everything resident.

The scheme is exactly equivalent to the unchunked computation, not an
approximation. See ``DESIGN.md`` for the derivation.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor

from .._model.kv_cache import KVCache

__all__ = [
    "chunked_set_transformer",
    "chunked_icl_encoder",
    "row_chunked",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COL_CHUNK_SIZE",
]

# Large enough to keep the GPU busy, small enough that (chunk x C x d) stays
# far below the activation that made the unchunked path OOM.
DEFAULT_CHUNK_SIZE = 8192

# Phase A runs the full stack over every context row, so its feed-forward
# intermediate scales with the column count. Slabbing columns bounds it.
DEFAULT_COL_CHUNK_SIZE = 32


def _fill_cache_from_hidden(block, hidden: Tensor, cache: KVCache, block_idx: int) -> None:
    """Store the K/V projections of ``hidden`` without running stage 2 over all rows.

    In ISAB stage 2 the call is ``multihead_attn2(src, hidden, hidden, need_kv=True)``.
    The returned projections are functions of ``hidden`` alone -- ``src`` only supplies
    the query. So a single dummy query row is enough to harvest them, which is what
    makes the phase A / phase B split cheap.
    """
    from .._model.kv_cache import KVCacheEntry

    *batch_shape, _, d_model = hidden.shape
    dummy_q = hidden.new_zeros((*batch_shape, 1, d_model))
    _, k_proj, v_proj = block.multihead_attn2(dummy_q, hidden, hidden, need_kv=True)
    cache.kv[block_idx] = KVCacheEntry(key=k_proj, value=v_proj)


def chunked_set_transformer(
    tf_col,
    src: Tensor,
    train_size: Optional[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    col_chunk_size: int = DEFAULT_COL_CHUNK_SIZE,
    inplace: bool = False,
) -> Tensor:
    """Run a ``SetTransformer`` with peak memory decoupled from the row count.

    Parameters
    ----------
    tf_col : SetTransformer
        The column-embedding set transformer (stack of ISAB blocks).

    src : Tensor
        Input of shape ``(..., T, d_model)``.

    train_size : Optional[int]
        Rows ``[0:train_size]`` form the inducing-point context. ``None`` means all
        rows do (the ``embed_with_test=True`` case), which is handled by treating
        the full sequence as the context.

    chunk_size : int, default=8192
        Rows streamed per phase-B step.

    col_chunk_size : int, default=32
        Column slabs processed per outer step. Bounds phase A, which would otherwise
        dominate peak memory.

    inplace : bool, default=False
        Overwrite ``src`` with the result instead of allocating an output tensor.
        Removes one full ``(N, C, d)`` allocation. Only safe when the caller does
        not need ``src`` afterwards.

    Returns
    -------
    Tensor
        Same shape as ``src``; numerically equivalent to ``tf_col(src, train_size)``.
    """
    n_rows = src.shape[-2]
    ctx = n_rows if train_size is None else train_size

    # `ISAB.forward` passes all-skip_value slices straight through. Batched training
    # uses this for padded columns; mirror it so the chunked path stays equivalent.
    skip_value = tf_col.blocks[0].skip_value
    skip_mask = (src == skip_value).all(dim=(-2, -1))
    if skip_mask.any():
        if skip_mask.all():
            return torch.full_like(src, skip_value)
        out = torch.empty_like(src)
        out[~skip_mask] = chunked_set_transformer(
            tf_col, src[~skip_mask], train_size, chunk_size, col_chunk_size, inplace
        )
        out[skip_mask] = skip_value
        return out

    # Chunking only the row axis is not enough. Phase A must run the whole stack over
    # every context row, and its feed-forward intermediate
    # (ctx x C x dim_feedforward) then becomes the new peak -- which is why TabPFN-3
    # also chunks phase (i) "along the (independent) column dimension".
    #
    # Columns really are independent here, so the cleanest form is a 2D sweep: take a
    # slab of columns, and run phase A then phase B entirely within it. Peak memory is
    # then bounded by (col_chunk x max(ctx, chunk_size) x d_ff) regardless of how many
    # rows or columns the dataset has.
    d_model = src.shape[-1]
    flat = src.reshape(-1, n_rows, d_model)  # collapse batch/column dims into one axis
    n_slabs = flat.shape[0]
    out_flat = flat if inplace else torch.empty_like(flat)

    for cstart in range(0, n_slabs, col_chunk_size):
        cstop = min(cstart + col_chunk_size, n_slabs)
        slab = flat[cstart:cstop]

        # ---- Phase A: inducing states over the context rows of this slab. ----
        # Block i+1's hidden depends on block i's output for context rows, so the
        # full stack runs here -- but only over ctx rows of this column slab.
        cache = KVCache()
        ctx_out = slab[:, :ctx, :]
        for block_idx, block in enumerate(tf_col.blocks):
            hidden = block.multihead_attn1(
                block.ind_vectors.expand(ctx_out.shape[0], block.num_inds, d_model),
                ctx_out,
                ctx_out,
            )
            _fill_cache_from_hidden(block, hidden, cache, block_idx)
            # Advance context rows to this block's output so the next block's stage 1
            # sees the same input it would in the unchunked path.
            ctx_out = block.multihead_attn2(ctx_out, cached_kv=cache.kv[block_idx])
            del hidden
        del ctx_out

        # ---- Phase B: stream this slab's rows against the cached K/V. ----
        # A row's stage-2 output depends only on that row and `hidden`, so row chunks
        # are independent. Write straight into the destination: a `torch.cat` would
        # transiently hold two full tensors and cap the saving at ~2x.
        #
        # Writing in place is safe because chunk i reads rows [i, i+chunk) and writes
        # only those same rows, and the first attention call already materialised a
        # new tensor before the write happens.
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = slab[:, start:stop, :]
            for block_idx, block in enumerate(tf_col.blocks):
                chunk = block.multihead_attn2(chunk, cached_kv=cache.kv[block_idx])
            out_flat[cstart:cstop, start:stop, :] = chunk
            del chunk
        del cache, slab

    return out_flat.reshape(src.shape)


@contextlib.contextmanager
def row_chunked(
    model,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    col_chunk_size: int = DEFAULT_COL_CHUNK_SIZE,
    enabled: bool = True,
    inplace: bool = True,
):
    """Temporarily route a model's column embedding through the chunked path.

    Parameters
    ----------
    model : TabICL
        A fitted TabICL backbone (``clf.model_``).

    chunk_size : int, default=8192
        Rows per phase-B chunk.

    col_chunk_size : int, default=32
        Columns per outer slab.

    enabled : bool, default=True
        When False this is a no-op, which keeps benchmark code branch-free.

    inplace : bool, default=True
        Let the chunked path overwrite its input buffer. Safe here because
        ``ColEmbedding`` does not reuse the tensor it hands to the set transformer.

    Examples
    --------
    >>> with row_chunked(clf.model_, chunk_size=4096):  # doctest: +SKIP
    ...     proba = clf.predict_proba(X_test)
    """
    if not enabled:
        yield model
        return

    col_embedder = model.col_embedder
    original = col_embedder.tf_col

    class _ChunkedProxy(torch.nn.Module):
        """Presents the SetTransformer interface, dispatching to the chunked path."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
            # The mixed-radix ensembling branch calls this repeatedly with cloned
            # buffers; chunking it is possible but unnecessary, so defer.
            return chunked_set_transformer(
                self.inner, src, train_size, chunk_size, col_chunk_size, inplace
            )

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    col_embedder.tf_col = _ChunkedProxy(original)
    try:
        yield model
    finally:
        col_embedder.tf_col = original


def chunked_icl_encoder(
    tf_icl,
    src: Tensor,
    train_size: Optional[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    inplace: bool = False,
) -> Tensor:
    """Run the ICL transformer with peak memory decoupled from the row count.

    Beyond roughly 150k rows the ICL stack, not the column embedder, owns the peak --
    measured at 9x its own input tensor, since every one of its 12 blocks carries a
    residual stream plus a feed-forward intermediate across all rows at once.

    The same observation that chunks the column embedder applies here. Within a block
    every row is a *query*, but only the first ``train_size`` rows supply keys and
    values, so a query row's output depends on that row and the shared train-derived
    K/V and nothing else. Harvest the K/V once per block, then stream queries.

    Two details make it exact rather than approximate:

    * The K/V are projected from the train rows *before* any row is overwritten, so
      writing results back in place cannot corrupt the context.
    * Train rows are queries too, and their outputs become the next block's context.
      Streaming them in the same pass keeps that correct, because the next block
      re-derives its K/V from the updated rows.

    Requires no positional encoding on the stack: rows in the ICL stage are
    exchangeable and ``tf_icl.rope`` is ``None``. A stack with rope would need the
    chunk's absolute offsets threading through, so this refuses rather than silently
    encoding the wrong positions.

    Parameters
    ----------
    tf_icl : Encoder
        The ICL transformer (``model.icl_predictor.tf_icl``).

    src : Tensor
        Row representations, shape ``(..., T, d_model)``.

    train_size : Optional[int]
        Rows ``[0:train_size]`` supply keys and values. ``None`` means all rows do.

    chunk_size : int, default=8192
        Query rows per step.

    inplace : bool, default=False
        Overwrite ``src`` rather than allocating an output buffer.

    Returns
    -------
    Tensor
        Same shape as ``src``; numerically equivalent to ``tf_icl(src, train_size)``.
    """
    if getattr(tf_icl, "rope", None) is not None:
        raise ValueError(
            "chunked_icl_encoder requires rope=None; a chunk would otherwise be "
            "encoded at the wrong absolute positions"
        )

    from .._model.kv_cache import KVCacheEntry

    n_rows = src.shape[-2]
    ctx_len = n_rows if train_size is None else train_size
    out = src if inplace else src.clone()

    for block in tf_icl.blocks:
        context = out[..., :ctx_len, :]
        # A single dummy query row is enough to project the context's K/V, exactly as
        # in the column embedder: the projections depend on the keys, not the query.
        _, k_proj, v_proj = block(q=out[..., :1, :], k=context, v=context, need_kv=True)
        cache = KVCacheEntry(key=k_proj, value=v_proj)

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            out[..., start:stop, :] = block(q=out[..., start:stop, :], cached_kv=cache)

        del cache, k_proj, v_proj

    return out
