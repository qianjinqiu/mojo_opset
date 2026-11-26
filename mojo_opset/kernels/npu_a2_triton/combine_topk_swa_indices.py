"""Top-k / sliding-window index concatenation for DeepSeek V4.1 sparse prefill (A2).

The upstream mapping (one program per request and worker, 256 workers per
request) is kept for every shape: the row-block grid and the whole-row store of
the A5 provider were measured on A5 only, so they are not used on this arch.
"""

import torch
import triton
import triton.language as tl

_COMBINE_TOPK_SWA_NUM_WORKERS = 256


@triton.jit
def _combine_emit_row(
    row,
    batch_idx,
    pos,
    gather_start,
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW: tl.constexpr,
):
    """Write the valid entries of one output row and its length."""
    # topk_len is fully determined by the query token's absolute position:
    # the indexer emits min((pos + 1) // compress_ratio, topk_tokens) valid
    # entries. Caller passes TOP_K=0 for SWA-only layers to zero this out.
    if COMPRESS_RATIO > 0:
        # Warmup batches can put pos below zero; keep the length >= 0.
        topk_len = tl.maximum(tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K), 0)
    else:
        # SWA-only layers pass TOP_K=0; skip the division entirely
        # (integer div by constexpr 0 is UB and yields garbage, which
        # turns the store index below negative).
        topk_len = 0
    # gather_len already excludes context below the request's replay
    # start (SWA bounded replay), so the window cannot start before
    # the gathered buffer does.
    swa_start = tl.maximum(tl.maximum(pos - (WINDOW_SIZE - 1), 0), gather_start)
    swa_len = tl.maximum(pos - swa_start + 1, 0)

    # Row offsets are int64 scalars: a large prefill batch times the
    # workspace row pitch can exceed 2^31.
    row64 = row.to(tl.int64)
    out_row = combined_indices_ptr + row64 * combined_indices_stride
    offset = tl.arange(0, PADDED_TOP_K)
    mask = offset < topk_len
    topk_indices = tl.load(topk_indices_ptr + row64 * topk_indices_stride + offset, mask=mask)
    # A failed (-1) or out-of-pool candidate stays -1 instead of landing
    # in the previous request's rows or this request's SWA rows.
    valid = (topk_indices >= 0) & (topk_indices < N)
    tl.store(
        out_row + offset,
        tl.where(valid, topk_indices + M * batch_idx, -1),
        mask=mask,
    )
    # Index into gathered buffer: N + (position - gather_start)
    # For positions [swa_start, pos], the buffer indices are:
    # [N + swa_start - gather_start, N + pos - gather_start]
    swa_offset = tl.arange(0, PADDED_WINDOW)
    tl.store(
        out_row + topk_len + swa_offset,
        M * batch_idx + N + swa_start + swa_offset - gather_start,
        mask=swa_offset < swa_len,
    )
    tl.store(combined_lens_ptr + row, topk_len + swa_len)


@triton.jit(do_not_specialize=["combined_indices_stride", "topk_indices_stride", "M", "N"])
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    # query_start_loc is a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    # The SWA portion of the gathered buffer starts from position
    # (seq_len - gather_len), not position 0. We need this offset
    # to correctly index into the gathered buffer.
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        # topk_len is fully determined by the query token's absolute position:
        # the indexer emits min((pos + 1) // compress_ratio, topk_tokens)
        # valid entries. Caller passes TOP_K=0 for SWA-only layers to zero
        # this out.
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        _combine_emit_row(
            token_idx,
            batch_idx,
            pos,
            gather_start,
            combined_indices_ptr,
            combined_indices_stride,
            combined_lens_ptr,
            topk_indices_ptr,
            topk_indices_stride,
            M,
            N,
            TOP_K,
            COMPRESS_RATIO,
            WINDOW_SIZE,
            PADDED_TOP_K,
            PADDED_WINDOW,
        )


def _combine_topk_swa_indices_impl(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
    combined_indices,
    combined_lens,
    owned,
):
    num_reqs = seq_lens.shape[0]
    if owned:
        combined_indices.fill_(-1)
    _combine_topk_swa_indices_kernel[(num_reqs, _COMBINE_TOPK_SWA_NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc.contiguous(),
        seq_lens.contiguous(),
        gather_lens.contiguous(),
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=max(triton.next_power_of_2(topk_indices.shape[-1]), 1),
        PADDED_WINDOW=max(triton.next_power_of_2(window_size), 1),
    )


@torch.library.custom_op(
    "mojo_npu_triton_a2::combine_topk_swa_indices_fwd",
    mutates_args=("combined_indices", "combined_lens"),
)
def combine_topk_swa_indices_fwd(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    owned: bool,
) -> None:
    _combine_topk_swa_indices_impl(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
        combined_indices,
        combined_lens,
        owned,
    )


@combine_topk_swa_indices_fwd.register_fake
def _combine_topk_swa_indices_fake(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
    combined_indices,
    combined_lens,
    owned,
) -> None:
    return None
