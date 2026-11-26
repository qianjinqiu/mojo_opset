"""Eager reference for the top-k / sliding-window index concatenation."""

import torch


def combine_topk_swa_indices_fwd(
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
    """Write the valid entries of every row and the row lengths, in place.

    An owned output is prefilled with ``-1``; a caller workspace keeps the
    entries it held outside the valid ranges.
    """
    if owned:
        combined_indices.fill_(-1)
    num_tokens, width = topk_indices.shape
    if num_tokens == 0:
        return None
    device = combined_indices.device

    qsl = query_start_loc.to(torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    req_ids = torch.repeat_interleave(torch.arange(seq_lens.shape[0], device=device), query_lens)
    starts = torch.repeat_interleave(qsl[:-1] - qsl[0], query_lens)
    positions = seq_lens.to(torch.int64)[req_ids] - query_lens[req_ids] + (torch.arange(num_tokens, device=device) - starts)
    if compress_ratio > 0:
        topk_lens = ((positions + 1) // compress_ratio).clamp(0, topk)
    else:
        topk_lens = torch.zeros_like(positions)
    gather_starts = seq_lens.to(torch.int64) - gather_lens.to(torch.int64)
    swa_starts = torch.maximum(
        torch.maximum(positions - (window_size - 1), torch.zeros_like(positions)),
        gather_starts[req_ids],
    )
    swa_lens = (positions - swa_starts + 1).clamp_min(0)

    # Candidates: only the first topk_len columns of each row are written.
    row_width = combined_indices.shape[1]
    candidate_width = min(width, row_width)
    columns = torch.arange(candidate_width, device=device)
    candidate_mask = columns.unsqueeze(0) < topk_lens.unsqueeze(-1)
    values = topk_indices[:, :candidate_width].to(torch.int64)
    valid = candidate_mask & (values >= 0) & (values < N)
    candidates = torch.where(valid, values + M * req_ids.unsqueeze(-1), -1)
    combined_indices[:, :candidate_width] = torch.where(
        candidate_mask, candidates, combined_indices[:, :candidate_width].to(torch.int64)
    ).to(torch.int32)

    # Window entries start at column topk_len of each row and may reach into
    # the candidate range when the row selected fewer than topk candidates.
    columns = torch.arange(row_width, device=device)
    window_mask = (columns.unsqueeze(0) >= topk_lens.unsqueeze(-1)) & (
        columns.unsqueeze(0) < (topk_lens + swa_lens).unsqueeze(-1)
    )
    window = M * req_ids.unsqueeze(-1) + N + swa_starts.unsqueeze(-1) + (
        columns.unsqueeze(0) - topk_lens.unsqueeze(-1)
    ) - gather_starts[req_ids].unsqueeze(-1)
    combined_indices.copy_(torch.where(window_mask, window, combined_indices.to(torch.int64)).to(torch.int32))
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return None
