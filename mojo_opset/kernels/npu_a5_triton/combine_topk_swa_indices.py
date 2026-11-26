"""Top-k / sliding-window index concatenation for DeepSeek V4.1 sparse prefill (A5).

Wide rows run at the HBM bound, so they keep the upstream mapping: one program
per (request, worker) walks that request's rows with a stride of 256. Two other
regimes pay a fixed cost per row that the data does not amortize, and each has
its own kernel behind a host-side shape rule:

* Narrow rows (at most ``_TILE_MAX_WIDTH`` wide): a per-row store costs one DMA
  handshake whatever the row width, and a row loop pays three of them per row.
  The tile kernel gives one program ``R`` contiguous rows, stores their rows as
  one unmasked block and their lengths as one vector, and resolves the requests
  from one vector read of the request table. The last tile slides back over rows
  the previous program covered; every value depends on the row alone, so the
  overlap writes the same bytes. A workspace row is read back only when some row
  of the tile leaves columns unwritten. ``R`` keeps about ``_TILE_MIN_PROGRAMS``
  programs in the launch, so that a small batch fits one wave on the cores.
* A batch with at most ``_TINY_ROWS`` rows per request: most of the 256 workers
  would be idle. One program per ``_RPP`` token rows is launched instead. The
  first row finds its request by a binary search over ``query_start_loc``; each
  later row walks forward from the previous row's request, because a scalar load
  of the table is uncached and a per-row search chains eleven of them.

An output owned by the entry is written whole by the tile kernel at any batch
size, and by the per-row kernels up to ``_WHOLE_ROW_MAX_ROWS`` rows, which spares
the ``-1`` prefill launch. Larger owned outputs keep the prefill: beyond that
size the extra tail stores cost more than the fill they replace.

A caller-provided workspace keeps its row tail: only the valid entries of each
row and every length are written.
"""

import torch
import triton
import triton.language as tl

_COMBINE_TOPK_SWA_NUM_WORKERS = 256
_TINY_ROWS = 8
_RPP = 8
# Wide rows: fewer rows per program. It halves the per-program length and, with
# fresh data on every call, the serialized row reads behind one another.
# Narrower rows keep more rows per program.
_RPP_WIDE = 4
_RPP_WIDE_MIN_ROW_WIDTH = 2048
_WHOLE_ROW_MAX_ROWS = 256
_TILE_MAX_WIDTH = 256
# A taller tile measured slower on the largest batches and overflows the UB
# beyond twice this size.
_TILE_MAX_ELEMS = 4096
# About this many programs at least, so that a small batch fits one wave.
_TILE_MIN_PROGRAMS = 28
# The tile kernel resolves requests with an [R, P] compare tile.
_TILE_MAX_REQUESTS = 1024
_INT32_MAX = 2**31 - 1


@triton.jit
def _combine_row_request(query_start_loc_ptr, base, row, num_reqs, SEARCH_STEPS: tl.constexpr):
    """The row's request: the last r with query_start_loc[r] - base <= row.

    A zero-length request is skipped: the row belongs to the next non-empty one.
    """
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), 1, dtype=tl.int32) * num_reqs
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        start = tl.load(query_start_loc_ptr + mid) - base
        go_right = start <= row
        lo = tl.where(go_right, mid, lo)
        hi = tl.where((lo < hi) & ~go_right, mid - 1, hi)
    return tl.minimum(lo, num_reqs - 1)


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
    WHOLE_ROW: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROW_HEAD: tl.constexpr,
    ROW_TAIL: tl.constexpr,
):
    """Write one output row and its length.

    ``WHOLE_ROW`` selects between two write sets with the same values:

    - false: only the ``topk_len`` candidate entries and the ``swa_len`` window
      entries are stored, so a caller-provided workspace keeps its row tail;
    - true: the whole ``ROW_WIDTH``-wide row is stored, the rest of it as ``-1``,
      so an output owned by the entry needs no fill before the launch.
    """
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
    topk_row = topk_indices_ptr + row64 * topk_indices_stride
    if WHOLE_ROW:
        # Index into gathered buffer: N + (position - gather_start)
        # For positions [swa_start, pos], the buffer indices are:
        # [N + swa_start - gather_start, N + pos - gather_start]
        head = tl.arange(0, ROW_HEAD)
        topk_indices = tl.load(topk_row + head, mask=head < topk_len)
        # A failed (-1) or out-of-pool candidate stays -1 instead of landing
        # in the previous request's rows or this request's SWA rows.
        valid = (topk_indices >= 0) & (topk_indices < N)
        window = M * batch_idx + N + swa_start + (head - topk_len) - gather_start
        values = tl.where(
            head < topk_len,
            tl.where(valid, topk_indices + M * batch_idx, -1),
            tl.where(head < topk_len + swa_len, window, -1),
        )
        if ROW_WIDTH <= ROW_HEAD:
            tl.store(out_row + head, values, mask=head < ROW_WIDTH)
        else:
            tl.store(out_row + head, values)
            tail_cols = ROW_HEAD + tl.arange(0, ROW_TAIL)
            tl.store(
                out_row + tail_cols,
                tl.where(
                    tail_cols < topk_len + swa_len,
                    M * batch_idx + N + swa_start + (tail_cols - topk_len) - gather_start,
                    -1,
                ),
                mask=tail_cols < ROW_WIDTH,
            )
    else:
        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_indices = tl.load(topk_row + offset, mask=mask)
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
    WHOLE_ROW: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROW_HEAD: tl.constexpr,
    ROW_TAIL: tl.constexpr,
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
            WHOLE_ROW,
            ROW_WIDTH,
            ROW_HEAD,
            ROW_TAIL,
        )


@triton.jit(do_not_specialize=["combined_indices_stride", "topk_indices_stride", "M", "N"])
def _combine_topk_swa_indices_rows(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    num_tokens,
    num_reqs,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW: tl.constexpr,
    WHOLE_ROW: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROW_HEAD: tl.constexpr,
    ROW_TAIL: tl.constexpr,
    RPP: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * RPP
    # query_start_loc is a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    base = tl.load(query_start_loc_ptr)
    total = tl.load(query_start_loc_ptr + num_reqs) - base

    # The request of the first row is found once; the rows after it only
    # move forward, so each one walks from the previous row's request. A
    # scalar load of the table is uncached and a per-row binary search
    # chains eleven of them.
    batch = _combine_row_request(query_start_loc_ptr, base, row0, num_reqs, SEARCH_STEPS)
    query_end = tl.load(query_start_loc_ptr + batch + 1) - base
    seq_len = tl.load(seq_lens_ptr + batch)
    gather_start = seq_len - tl.load(gather_lens_ptr + batch)

    for i in tl.static_range(RPP):
        row = row0 + i
        if row < total:
            # Advance past the requests that end at or before this row; a
            # zero-length request ends where it starts and is skipped too.
            while row >= query_end:
                batch = batch + 1
                query_end = tl.load(query_start_loc_ptr + batch + 1) - base
                seq_len = tl.load(seq_lens_ptr + batch)
                gather_start = seq_len - tl.load(gather_lens_ptr + batch)
            _combine_emit_row(
                row, batch, seq_len - query_end + row, gather_start,
                combined_indices_ptr, combined_indices_stride, combined_lens_ptr,
                topk_indices_ptr, topk_indices_stride, M, N,
                TOP_K, COMPRESS_RATIO, WINDOW_SIZE, PADDED_TOP_K, PADDED_WINDOW,
                WHOLE_ROW, ROW_WIDTH, ROW_HEAD, ROW_TAIL,
            )
        elif row < num_tokens:
            # Rows past the last request's queries belong to no request.
            tl.store(combined_lens_ptr + row, 0)


@triton.jit
def _combine_tile_rows(
    row0,
    rows,
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    num_reqs,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    WHOLE_ROW: tl.constexpr,
):
    """Build and store the ``[R, ROW_WIDTH]`` tile of the ``R`` rows ``rows``.

    The request of every row is resolved without a per-row scalar load: the
    request table is read once as ``[P]`` vectors, and each row counts the
    non-empty requests that start at or before it; the request is the last of
    them, so an empty request in between is skipped.

    The tile is stored as one unmasked flat block and the lengths as one
    unmasked vector: two write handshakes for ``R`` rows.
    """
    # query_start_loc is a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    if P == 1:
        # One request: every row belongs to it, and its metadata is scalar.
        base = tl.load(query_start_loc_ptr)
        query_end = tl.load(query_start_loc_ptr + 1) - base
        seq_len = tl.load(seq_lens_ptr)
        gather_start = seq_len - tl.load(gather_lens_ptr)
        req_term = tl.zeros_like(rows)
    else:
        idx = tl.arange(0, P)
        has_req = idx < num_reqs
        base = tl.load(query_start_loc_ptr)
        start_all = tl.load(query_start_loc_ptr + idx, mask=has_req, other=2147483647) - base
        end_all = tl.load(query_start_loc_ptr + idx + 1, mask=has_req, other=2147483647) - base
        seq_all = tl.load(seq_lens_ptr + idx, mask=has_req, other=0)
        gather_all = tl.load(gather_lens_ptr + idx, mask=has_req, other=0)

        # A zero-length request is skipped: the row belongs to the next
        # non-empty one.
        in_request = (
            has_req[None, :] & (start_all[None, :] <= rows[:, None]) & (start_all[None, :] < end_all[None, :])
        )
        req = tl.max(tl.where(in_request, idx[None, :], -1), axis=1)
        one_hot = req[:, None] == idx[None, :]
        query_end = tl.sum(tl.where(one_hot, end_all[None, :], 0), axis=1)
        seq_len = tl.sum(tl.where(one_hot, seq_all[None, :], 0), axis=1)
        gather_start = seq_len - tl.sum(tl.where(one_hot, gather_all[None, :], 0), axis=1)
        req_term = M * req

    pos = seq_len - query_end + rows
    if COMPRESS_RATIO > 0:
        # Warmup batches can put pos below zero; keep the length >= 0.
        topk_len = tl.maximum(tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K), 0)
    else:
        # SWA-only layers pass TOP_K=0: no candidate is ever valid.
        topk_len = tl.zeros_like(pos)
    # gather_len already excludes context below the request's replay
    # start, so the window cannot start before the gathered buffer does.
    swa_start = tl.maximum(tl.maximum(pos - (WINDOW_SIZE - 1), 0), gather_start)
    swa_len = tl.maximum(pos - swa_start + 1, 0)

    cols = tl.arange(0, ROW_WIDTH)
    column = cols[None, :]
    # Index into the gathered buffer: N + (position - gather_start).
    window = (req_term + N + swa_start - topk_len - gather_start)[:, None] + column
    in_window = (column >= topk_len[:, None]) & (column < (topk_len + swa_len)[:, None])

    tile_ptr = combined_indices_ptr + rows[:, None] * combined_indices_stride + column
    if WHOLE_ROW:
        # Every column of an owned row is defined: the rest of it is -1.
        tile = tl.where(in_window, window, -1)
    else:
        # A caller workspace keeps its content outside the written entries.
        # Its rows are read back only when some row of the tile leaves
        # columns unwritten: a full window covers the whole row.
        covered = (topk_len == 0) & (topk_len + swa_len >= ROW_WIDTH)
        if tl.min(covered.to(tl.int32), axis=0) == 1:
            tile = window
        else:
            tile = tl.where(in_window, window, tl.load(tile_ptr))
    if TOP_K > 0:
        cand_ptr = topk_indices_ptr + rows[:, None] * topk_indices_stride + column
        is_candidate = column < topk_len[:, None]
        candidates = tl.load(cand_ptr, mask=is_candidate, other=0)
        # A failed (-1) or out-of-pool candidate stays -1 instead of landing
        # in the previous request's rows or this request's SWA rows.
        kept = (candidates >= 0) & (candidates < N)
        tile = tl.where(is_candidate, tl.where(kept, candidates + req_term[:, None], -1), tile)

    # One block for the whole tile: the rows are contiguous (row stride equals
    # the row width) and all R of them exist.
    flat = tl.reshape(tile, (R * ROW_WIDTH,))
    tl.store(combined_indices_ptr + row0 * combined_indices_stride + tl.arange(0, R * ROW_WIDTH), flat)
    tl.store(combined_lens_ptr + row0 + tl.arange(0, R), topk_len + swa_len)


@triton.jit(do_not_specialize=["combined_indices_stride", "topk_indices_stride", "M", "N"])
def _combine_topk_swa_indices_tile(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    num_tokens,
    num_reqs,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    WHOLE_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    # Full tiles only (R <= num_tokens). The last tile slides back over rows
    # the previous program already covered; every value depends on the row
    # alone, so the overlap writes the same bytes twice.
    row0 = tl.minimum(pid * R, num_tokens - R)
    _combine_tile_rows(
        row0, row0 + tl.arange(0, R), combined_indices_ptr, combined_indices_stride, combined_lens_ptr,
        topk_indices_ptr, topk_indices_stride, query_start_loc_ptr, seq_lens_ptr, gather_lens_ptr,
        num_reqs, M, N, TOP_K, COMPRESS_RATIO, WINDOW_SIZE, ROW_WIDTH, R, P, WHOLE_ROW,
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
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    row_width = combined_indices.shape[1]
    requests_pad = triton.next_power_of_2(max(num_reqs, 1))
    tile = (
        row_width <= _TILE_MAX_WIDTH
        # The tile columns are the whole row.
        and row_width & (row_width - 1) == 0
        and requests_pad <= _TILE_MAX_REQUESTS
        # Row offsets stay int32.
        and num_tokens * row_width <= _INT32_MAX
        # A workspace qualifies only when its rows are contiguous: the tile is
        # stored as one flat block.
        and (owned or (combined_indices.stride(0) == row_width and combined_lens.stride(0) == 1))
    )
    whole_row = owned and (tile or num_tokens <= _WHOLE_ROW_MAX_ROWS)
    if owned and not whole_row:
        combined_indices.fill_(-1)

    width = topk_indices.shape[-1]
    # Whole-row store tiles: the head covers the candidate region (and the
    # whole row when it is narrower than the candidate buffer), the tail the
    # rest of the row.
    row_head = max(triton.next_power_of_2(min(width, row_width)), 1)
    launch = dict(
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        WHOLE_ROW=whole_row,
        ROW_WIDTH=row_width,
    )
    query_start_loc = query_start_loc.contiguous()
    seq_lens = seq_lens.contiguous()
    gather_lens = gather_lens.contiguous()
    if tile:
        # Never more rows than the batch holds.
        r = max(1, min(_TILE_MAX_ELEMS // row_width, num_tokens // _TILE_MIN_PROGRAMS, num_tokens))
        _combine_topk_swa_indices_tile[((num_tokens + r - 1) // r,)](
            combined_indices,
            combined_indices.stride(0),
            combined_lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            num_tokens,
            num_reqs,
            M,
            N,
            R=r,
            P=requests_pad,
            **launch,
        )
    elif num_tokens <= num_reqs * _TINY_ROWS:
        rpp = _RPP_WIDE if row_width >= _RPP_WIDE_MIN_ROW_WIDTH else _RPP
        per_row = dict(
            PADDED_TOP_K=max(triton.next_power_of_2(width), 1),
            PADDED_WINDOW=max(triton.next_power_of_2(window_size), 1),
            ROW_HEAD=row_head,
            ROW_TAIL=max(triton.next_power_of_2(max(row_width - row_head, 0)), 1),
        )
        _combine_topk_swa_indices_rows[((num_tokens + rpp - 1) // rpp,)](
            combined_indices,
            combined_indices.stride(0),
            combined_lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            num_tokens,
            num_reqs,
            M,
            N,
            RPP=rpp,
            SEARCH_STEPS=max(num_reqs, 1).bit_length(),
            **launch,
            **per_row,
        )
    else:
        _combine_topk_swa_indices_kernel[(num_reqs, _COMBINE_TOPK_SWA_NUM_WORKERS)](
            combined_indices,
            combined_indices.stride(0),
            combined_lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            **launch,
            PADDED_TOP_K=max(triton.next_power_of_2(width), 1),
            PADDED_WINDOW=max(triton.next_power_of_2(window_size), 1),
            ROW_HEAD=row_head,
            ROW_TAIL=max(triton.next_power_of_2(max(row_width - row_head, 0)), 1),
        )


@torch.library.custom_op(
    "mojo_npu_triton_a5::combine_topk_swa_indices_fwd",
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
