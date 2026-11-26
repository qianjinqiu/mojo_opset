"""Top-k / sliding-window index concatenation for DeepSeek V4.1 sparse prefill."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

# FlashMLA sparse prefill requires the combined width to be a multiple of 128.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128
_INT32_MAX = 2**31 - 1


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate each row's top-k indices with its sliding-window indices.

    ``combined_indices`` is int32 ``[num_tokens, round_up(topk + window_size, 128)]``
    and ``combined_lens`` int32 ``[num_tokens]``. Row ``t`` holds, for the token
    at absolute position ``pos``:

    * ``topk_len = clamp(min((pos + 1) // compress_ratio, topk), 0)`` candidate
      entries taken from ``topk_indices[t]``; a candidate below 0 or at least
      ``N`` is stored as ``-1``, any other as ``candidate + M * request``;
    * then ``swa_len`` window entries ``M * request + N + position -
      gather_start``, one for each position from
      ``max(pos - (window_size - 1), 0, gather_start)`` to ``pos``.

    ``combined_lens[t] = topk_len + swa_len``. ``compress_ratio = 0`` selects a
    sliding-window-only layer (``topk_len = 0``). Every combined index is below
    ``num_reqs * M``.

    ``topk_indices`` is int32 ``[num_tokens, width]`` with unit column stride
    and ``topk <= width``; ``query_start_loc`` is the int32 cu_seqlens of the
    chunk (``[num_reqs + 1]``, any base value, the first entry is subtracted);
    ``seq_lens`` and ``gather_lens`` are int32 ``[num_reqs]``. ``num_tokens``
    equals ``query_start_loc[num_reqs] - query_start_loc[0]`` and
    ``N + gather_lens <= M`` are preconditions of the caller.

    ``out`` may reuse a persistent ``(indices, lens)`` workspace of exactly
    those shapes; only the valid entries of each row are written and the rest
    of the row keeps its content. Without ``out`` the buffers are allocated and
    every entry is defined. ``num_reqs * M`` must not exceed ``2**31``: the
    combined indices are int32.
    """
    _require_inference("combine_topk_swa_indices", topk_indices, query_start_loc, seq_lens, gather_lens)
    if topk_indices.ndim != 2 or topk_indices.dtype != torch.int32 or topk_indices.stride(1) != 1:
        raise ValueError("topk_indices must be an int32 [num_tokens, width] tensor with unit column stride")
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.numel()
    if query_start_loc.ndim != 1 or query_start_loc.dtype != torch.int32:
        raise ValueError("query_start_loc must be a 1D int32 tensor")
    if query_start_loc.numel() != num_reqs + 1:
        raise ValueError("query_start_loc must hold num_reqs + 1 entries")
    for name, tensor in (("seq_lens", seq_lens), ("gather_lens", gather_lens)):
        if tensor.ndim != 1 or tensor.numel() != num_reqs or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be a 1D int32 tensor of length num_reqs")
        if tensor.device != topk_indices.device:
            raise ValueError(f"{name} must be on the topk_indices device")
    if query_start_loc.device != topk_indices.device:
        raise ValueError("query_start_loc must be on the topk_indices device")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if compress_ratio < 0:
        raise ValueError("compress_ratio must be non-negative")
    if topk < 0 or topk > topk_indices.shape[1]:
        raise ValueError("topk must lie in [0, width]")
    if M < 0 or N < 0:
        raise ValueError("M and N must be non-negative")
    if num_reqs * M > _INT32_MAX + 1:
        raise ValueError("num_reqs * M must not exceed 2**31")

    row_width = -(-(topk + window_size) // _SPARSE_PREFILL_TOPK_ALIGNMENT) * _SPARSE_PREFILL_TOPK_ALIGNMENT
    if out is None:
        # Every entry is written: whole rows by the kernel or a -1 prefill.
        combined_indices = torch.empty((num_tokens, row_width), dtype=torch.int32, device=topk_indices.device)
        combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
        owned = True
    else:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ValueError("out must be an (indices, lens) pair")
        combined_indices, combined_lens = out
        if (
            combined_indices.ndim != 2
            or combined_indices.dtype != torch.int32
            or tuple(combined_indices.shape) != (num_tokens, row_width)
            or combined_indices.stride(1) != 1
        ):
            raise ValueError("out indices must be an int32 [num_tokens, round_up(topk + window_size, 128)] tensor")
        if (
            combined_lens.ndim != 1
            or combined_lens.dtype != torch.int32
            or combined_lens.numel() != num_tokens
            or not combined_lens.is_contiguous()
        ):
            raise ValueError("out lens must be a contiguous int32 [num_tokens] tensor")
        if combined_indices.device != topk_indices.device or combined_lens.device != topk_indices.device:
            raise ValueError("out must be on the topk_indices device")
        owned = False

    if num_tokens == 0:
        return combined_indices, combined_lens
    forward, _ = load_impl("combine_topk_swa_indices", implementation)
    forward(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        int(window_size),
        int(compress_ratio),
        int(topk),
        int(M),
        int(N),
        combined_indices,
        combined_lens,
        owned,
    )
    return combined_indices, combined_lens
