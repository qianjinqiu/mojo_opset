"""Tests for functions.combine_topk_swa_indices.

The op is integer-only (out of scope of docs/operator_accuracy_standard.md §1,
which covers floating-point compute), so every case pins the result bit-exactly
against the torch reference with ``torch.equal``.
"""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

WINDOW = 128
TOPK = 2048
WIDTH = 2048
ALIGN = 128
GARBAGE = 0x5A5A5A5A

# (query_lens, seq_lens, compress_ratio, topk, width, N): production shapes
# (one prefill chunk of up to four requests, index_topk 2048) and the shapes
# that exercise each kernel. Rows up to 256 wide take the tile kernel: its
# request tile is 1 wide for one request and 256 wide for the decode-like batch,
# and the tile height leaves an overlap in the last program unless stated.
SHAPES = [
    ([8192], [8192], 2, 2048, 2048, 4096),  # prefill chunk, one request
    ([8192], [8192], 1, 2048, 2048, 8192),  # ratio-1 layer
    ([2048] * 4, [2048] * 4, 2, 2048, 2048, 1024),  # four requests per chunk
    ([8192], [16384], 2, 2048, 2048, 8192),  # continuation chunk
    ([2048], [2048], 2, 512, 512, 1024),  # smaller index_topk
    ([8192], [8192], 0, 0, 2048, 0),  # sliding-window-only layer
    ([1] * 256, [4096] * 256, 0, 0, 0, 0),  # sliding-window-only decode batch
    ([2048], [4096], 0, 0, 0, 0),  # sliding-window-only continuation chunk
    ([200], [200], 2, 64, 64, 100),  # row 128 with candidates, 200 rows
    ([137, 63], [300, 63], 2, 100, 128, 150),  # row 256 with candidates, two requests
    ([500], [500], 2, 128, 128, 64),  # row 256, 500 rows
    ([1] * 256, [4096] * 256, 2, 512, 512, 2048),  # row 640: row-block kernel, 8 rows
    ([64], [64], 2, 2048, 2048, 32),  # short prompt
    ([1], [1], 2, 2048, 2048, 1),  # single token
]

TINY_ROWS = ([1] * 256, [4096] * 256, 2, 2048, 2048, 2048)


def _row_width(topk, window_size=WINDOW):
    return -(-(topk + window_size) // ALIGN) * ALIGN


def _gather(seq_lens, query_lens, window_size=WINDOW):
    return [min(s, q + window_size - 1) for s, q in zip(seq_lens, query_lens)]


def _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device, seed=0, base=0,
            gather=None, invalid=None, window_size=WINDOW):
    torch.manual_seed(seed)
    gather = _gather(seq_lens, query_lens, window_size) if gather is None else gather
    m = n_candidates + max(gather, default=0)
    query_start_loc = [base]
    for q in query_lens:
        query_start_loc.append(query_start_loc[-1] + q)
    topk_indices = torch.randint(
        0, max(n_candidates, 1), (sum(query_lens), width), dtype=torch.int32, device=device
    )
    if invalid == "failed":
        topk_indices[:, ::7] = -1
    elif invalid == "out_of_pool":
        topk_indices[:, 1::5] = n_candidates + 3
        topk_indices[:, 3::11] = 2 * n_candidates + 1
    elif invalid == "negative":
        topk_indices[:, 1::5] = -5
    return (
        topk_indices,
        torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(gather, dtype=torch.int32, device=device),
        window_size,
        compress_ratio,
        topk,
        m,
        n_candidates,
    )


def _run(implementation, args, out=None):
    """Run the candidate and the reference from equal starting buffers."""
    candidate_out = None
    reference_out = None
    if out is not None:
        candidate_out = (out[0].clone(), out[1].clone())
        reference_out = (out[0].clone(), out[1].clone())
    actual = F.combine_topk_swa_indices(*args, out=candidate_out, implementation=implementation)
    expected = F.combine_topk_swa_indices(*args, out=reference_out, implementation="torch")
    assert torch.equal(actual[0].cpu(), expected[0].cpu()), "combined_indices differ"
    assert torch.equal(actual[1].cpu(), expected[1].cpu()), "combined_lens differ"
    if candidate_out is not None:
        # A workspace is written in place and returned as is.
        assert actual[0].data_ptr() == candidate_out[0].data_ptr()
        assert actual[1].data_ptr() == candidate_out[1].data_ptr()
    return actual


def _workspace(num_tokens, topk, device, window_size=WINDOW):
    row_width = _row_width(topk, window_size)
    return (
        torch.full((num_tokens, row_width), GARBAGE, dtype=torch.int32, device=device),
        torch.full((num_tokens,), GARBAGE, dtype=torch.int32, device=device),
    )


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
@pytest.mark.parametrize("owned", [True, False], ids=["owned", "workspace"])
@pytest.mark.parametrize(
    "query_lens,seq_lens,compress_ratio,topk,width,n_candidates",
    SHAPES + [TINY_ROWS],
    ids=[
        "prefill_8192_cr2", "prefill_8192_cr1", "chunk_4x2048_cr2", "chunked_16k_8k_cr2",
        "prefill_2048_topk512", "swa_only_8192", "swa_decode_256x1", "swa_chunked_4k_2k",
        "tile_200_topk64", "tile_two_req_topk100", "tile_500_topk128", "rows_256x1_topk512",
        "small_64", "small_1", "decode_mixed_256x1",
    ],
)
def test_combine_topk_swa_indices_shapes(accuracy_backend, owned, query_lens, seq_lens, compress_ratio,
                                         topk, width, n_candidates):
    """Production and boundary shapes through the owned output and the workspace."""
    implementation, _, device = accuracy_backend
    args = _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device)
    out = None if owned else _workspace(args[0].shape[0], topk, device)
    indices, lens = _run(implementation, args, out)
    if owned:
        assert indices.shape == (args[0].shape[0], _row_width(topk)) and indices.dtype == torch.int32
        assert lens.shape == (args[0].shape[0],) and lens.dtype == torch.int32


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "query_lens,seq_lens,compress_ratio,topk,width,n_candidates,base,gather,invalid",
    [
        ([64], [32], 2, 512, 512, 4096, 0, None, None),  # seq_len below query_len: negative positions
        ([32, 32], [0, 0], 2, 512, 512, 4096, 0, None, None),  # warmup batch: every position negative
        ([4, 4], [4, 4], 0, 0, 1, 4096, 0, None, None),  # compress_ratio 0 with a zero topk
        ([32, 64], [48, 80], 2, 512, 512, 4096, 32, None, None),  # nonzero query_start_loc base
        ([32, 64], [48, 80], 2, 512, 512, 4096, 0, None, "failed"),
        ([32, 64], [48, 80], 2, 512, 512, 4096, 0, None, "out_of_pool"),
        ([32, 64], [48, 80], 2, 512, 512, 4096, 0, None, "negative"),
        ([24, 12], [40, 12], 2, 16, 16, 20, 0, [24, 12], None),  # replay floor
        ([20, 12], [40, 12], 1, 0, 1, 40, 0, [24, 12], None),  # replay floor, window-only
        ([5, 0, 4], [100, 50, 60], 2, 512, 512, 50, 0, None, None),  # zero-length request
        ([2048], [2048], 2, 512, 2048, 1024, 0, None, None),  # topk below the buffer width
        ([2048], [2048], 2, 1000, 1000, 1024, 0, None, None),  # topk that is not a power of two
        ([64], [64], 0, 0, 0, 0, 0, None, None),  # zero-width top-k buffer
        ([1, 0, 1, 1, 0, 2, 1], [50, 3, 40, 9, 0, 12, 300], 2, 2048, 2048, 4096, 17, None, "negative"),
        ([1] * 40 + [8] * 3, [900] * 40 + [8, 300, 40], 2, 512, 512, 4096, 5, None, None),
        ([3, 0, 5, 7], [9, 4, 5, 40], 0, 0, 0, 79, 17, None, None),
        ([24, 12], [40, 12], 2, 16, 16, 20, 0, [24, 12], "out_of_pool"),
        ([32, 64], [48, 80], 2, 64, 64, 4096, 0, None, "failed"),
        ([500], [900], 1, 128, 128, 64, 0, None, "out_of_pool"),
    ],
    ids=[
        "seq_len_below_query_len", "seq_len_zero", "compress_ratio_zero", "nonzero_base", "failed",
        "out_of_pool", "negative", "replay_floor", "replay_floor_topk0", "zero_len_request",
        "topk_below_width", "nonpow2_topk", "zero_width", "walk_zero_len_many", "walk_mixed_batch",
        "tile_zero_len_requests", "tile_replay_floor_topk16", "tile_failed_candidates",
        "tile_500_out_of_pool_cr1",
    ],
)
def test_combine_topk_swa_indices_edge_cases(accuracy_backend, query_lens, seq_lens, compress_ratio, topk,
                                             width, n_candidates, base, gather, invalid):
    """The upstream edge cases: clamps, rebasing, candidate filtering, replay floor."""
    implementation, _, device = accuracy_backend
    args = _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device, base=base,
                   gather=gather, invalid=invalid)
    out = _workspace(args[0].shape[0], topk, device)
    _run(implementation, args, out)


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
@pytest.mark.parametrize("window_size", [8, 64, 256])
def test_combine_topk_swa_indices_window_sizes(accuracy_backend, window_size):
    implementation, _, device = accuracy_backend
    args = _inputs([512], [512], 2, 512, 512, 1024, device, window_size=window_size)
    _run(implementation, args, _workspace(512, 512, device, window_size))


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
def test_combine_topk_swa_indices_workspace_tail_preserved(accuracy_backend):
    """Only the valid entries of a row are written; the rest of the row keeps its content."""
    implementation, _, device = accuracy_backend
    args = _inputs([1024, 1024], [4096, 2048], 2, 2048, 2048, 2048, device)
    workspace = _workspace(2048, 2048, device)
    indices, lens = _run(implementation, args, workspace)
    columns = torch.arange(indices.shape[1], device=device)
    written = columns.unsqueeze(0) < lens.unsqueeze(-1)
    assert written.numel() - written.sum() > 0
    assert (indices[~written] == GARBAGE).all()


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
def test_combine_topk_swa_indices_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): no rows, no stores."""
    implementation, _, device = accuracy_backend
    args = _inputs([], [], 2, 2048, 2048, 0, device)
    indices, lens = _run(implementation, args)
    assert indices.shape == (0, _row_width(2048)) and indices.numel() == 0
    assert lens.shape == (0,) and lens.numel() == 0
    workspace = _workspace(0, 2048, device)
    indices, lens = F.combine_topk_swa_indices(*args, out=workspace, implementation=implementation)
    assert indices.data_ptr() == workspace[0].data_ptr() and lens.numel() == 0


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
def test_combine_topk_swa_indices_semantics(accuracy_backend):
    """Row layout: candidates, then window entries; failed candidates stay -1."""
    implementation, _, device = accuracy_backend
    query_lens, seq_lens, compress_ratio, topk, width, n_candidates = [64], [64], 2, 32, 32, 32
    args = _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device, invalid="failed")
    indices, lens = F.combine_topk_swa_indices(*args, implementation=implementation)
    topk_indices, query_start_loc, seq_lens_t, gather_lens_t = args[:4]
    for token in (0, 1, 5, 30, 63):
        position = token
        topk_len = min((position + 1) // compress_ratio, topk)
        swa_start = max(position - (WINDOW - 1), 0, seq_lens[0] - int(gather_lens_t[0]))
        swa_len = position - swa_start + 1
        assert int(lens[token]) == topk_len + swa_len
        candidates = topk_indices[token, :topk_len].to(torch.int64)
        expected = torch.where(candidates >= 0, candidates, torch.tensor(-1, dtype=torch.int64, device=device))
        assert torch.equal(indices[token, :topk_len].to(torch.int64), expected)
        window = torch.arange(swa_start, position + 1, device=device) - (seq_lens[0] - int(gather_lens_t[0]))
        assert torch.equal(indices[token, topk_len:topk_len + swa_len].to(torch.int64),
                           n_candidates + window)
        assert (indices[token, topk_len + swa_len:] == -1).all()


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
def test_combine_topk_swa_indices_largest_request_term(accuracy_backend):
    """M at the int32 limit of the contract: the request term of the last request stays exact."""
    implementation, _, device = accuracy_backend
    num_reqs, n_candidates = 256, 2048
    args = list(_inputs([1] * num_reqs, [4096] * num_reqs, 2, 2048, 2048, n_candidates, device))
    args[7] = (2**31) // num_reqs
    indices, lens = _run(implementation, args)
    assert int(lens[255]) == 2048 + WINDOW
    # Request 255: M * 255 + N + (position - gather_start) for the last of the 128 window positions.
    assert int(indices[255, -1]) == args[7] * (num_reqs - 1) + n_candidates + WINDOW - 1


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "broken,match",
    [
        ("topk_indices_dtype", "topk_indices must be"),
        ("topk_indices_stride", "topk_indices must be"),
        ("query_start_loc_dtype", "query_start_loc must be"),
        ("query_start_loc_length", "query_start_loc must hold"),
        ("seq_lens_dtype", "seq_lens must be"),
        ("gather_lens_length", "gather_lens must be"),
        ("query_start_loc_device", "must be on the topk_indices device"),
        ("window_size", "window_size must be positive"),
        ("compress_ratio", "compress_ratio must be non-negative"),
        ("topk_width", "topk must lie in"),
        ("m_negative", "M and N must be non-negative"),
        ("index_overflow", "num_reqs \\* M must not exceed"),
        ("out_pair", "out must be an \\(indices, lens\\) pair"),
        ("out_indices_shape", "out indices must be"),
        ("out_indices_stride", "out indices must be"),
        ("out_lens_stride", "out lens must be"),
        ("out_lens_device", "out must be on the topk_indices device"),
    ],
)
def test_combine_topk_swa_indices_rejects_invalid_inputs(accuracy_backend, broken, match):
    """Each contract check raises before any kernel is launched."""
    implementation, _, device = accuracy_backend
    args = list(_inputs([64, 64], [64, 64], 2, 512, 512, 1024, device))
    num_tokens = args[0].shape[0]
    out = _workspace(num_tokens, 512, device)
    if broken == "topk_indices_dtype":
        args[0] = args[0].to(torch.int64)
    elif broken == "topk_indices_stride":
        args[0] = torch.zeros(num_tokens, 1024, dtype=torch.int32, device=device)[:, ::2]
    elif broken == "query_start_loc_dtype":
        args[1] = args[1].to(torch.int64)
    elif broken == "query_start_loc_length":
        args[1] = args[1][:1]
    elif broken == "seq_lens_dtype":
        args[2] = args[2].to(torch.int64)
    elif broken == "gather_lens_length":
        args[3] = args[3][:1]
    elif broken == "query_start_loc_device":
        args[1] = args[1].cpu()
    elif broken == "window_size":
        args[4] = 0
    elif broken == "compress_ratio":
        args[5] = -1
    elif broken == "topk_width":
        args[6] = 513
    elif broken == "m_negative":
        args[7] = -1
    elif broken == "index_overflow":
        args[7] = 2**30 + 1
    elif broken == "out_pair":
        out = out[0]
    elif broken == "out_indices_shape":
        out = (torch.zeros(num_tokens, 768, dtype=torch.int32, device=device), out[1])
    elif broken == "out_indices_stride":
        wide = torch.zeros(num_tokens, 1280, dtype=torch.int32, device=device)
        out = (wide[:, ::2], out[1])
    elif broken == "out_lens_stride":
        wide = torch.zeros(2 * num_tokens, dtype=torch.int32, device=device)
        out = (out[0], wide[::2])
    elif broken == "out_lens_device":
        out = (out[0], out[1].cpu())
    with pytest.raises(ValueError, match=match):
        F.combine_topk_swa_indices(*args, out=out, implementation=implementation)


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
@pytest.mark.parametrize("query_lens,seq_lens,compress_ratio,topk,width,n_candidates",
                         [([2048], [2048], 0, 0, 0, 0), ([200], [200], 2, 64, 64, 100)],
                         ids=["swa", "with_candidates"])
def test_combine_topk_swa_indices_padded_workspace(accuracy_backend, query_lens, seq_lens, compress_ratio, topk,
                                                   width, n_candidates):
    """A workspace whose row pitch exceeds the row width: the tile kernel needs contiguous
    rows, so the per-row kernels write it; the padding keeps its content."""
    implementation, _, device = accuracy_backend
    args = _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device)
    num_tokens = sum(query_lens)
    row_width = _row_width(topk)
    padded = torch.full((num_tokens, row_width + 64), GARBAGE, dtype=torch.int32, device=device)
    workspace = (padded[:, :row_width], torch.full((num_tokens,), GARBAGE, dtype=torch.int32, device=device))
    assert workspace[0].stride(0) == row_width + 64
    _run(implementation, args, workspace)
    assert (padded[:, row_width:] == GARBAGE).all()


@pytest.mark.api("functions.combine_topk_swa_indices")
def test_combine_topk_swa_indices_launch_rules(accuracy_backend, monkeypatch):
    """Row width and rows per request select the kernel; the tile height and the
    row-block height follow the calibrated rules."""
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("A5 Triton dispatch")
    preload("combine_topk_swa_indices", implementation="triton", target=target)
    from mojo_opset.kernels.npu_a5_triton import combine_topk_swa_indices as provider

    launches = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, tuple(grid), kwargs.get("R", kwargs.get("RPP")), kwargs.get("P")))

            return launch

    for name in ("tile", "rows", "kernel"):
        monkeypatch.setattr(provider, f"_combine_topk_swa_indices_{name}", _Recorder(name))

    def _padded(num_tokens, row_width):
        wide = torch.full((num_tokens, row_width + 64), GARBAGE, dtype=torch.int32, device=device)
        return wide[:, :row_width], torch.full((num_tokens,), GARBAGE, dtype=torch.int32, device=device)

    # (query_lens, topk, compress_ratio, output, launch): kernel, grid, tile height or
    # rows per program, request tile width.
    CASES = [
        ([8192], 0, 0, "owned", ("tile", (256,), 32, 1)),
        ([64], 0, 0, "owned", ("tile", (32,), 2, 1)),
        ([1] * 256, 0, 0, "workspace", ("tile", (29,), 9, 256)),
        ([2048], 64, 2, "owned", ("tile", (128,), 16, 1)),
        ([2048], 64, 2, "padded", ("kernel", (1, 256), None, None)),
        ([1] * 256, 2048, 2, "workspace", ("rows", (64,), 4, None)),
        ([1] * 256, 512, 2, "workspace", ("rows", (32,), 8, None)),
        ([1], 2048, 2, "owned", ("rows", (1,), 4, None)),
        ([8192], 2048, 2, "workspace", ("kernel", (1, 256), None, None)),
    ]
    for query_lens, topk, compress_ratio, output, expected in CASES:
        seq_lens = [q if q > 1 else 4096 for q in query_lens]
        n_candidates = 0 if topk == 0 else max(seq_lens) // compress_ratio
        args = _inputs(query_lens, seq_lens, compress_ratio, topk, topk, n_candidates, device)
        num_tokens, row_width = sum(query_lens), _row_width(topk)
        out = {
            "owned": None,
            "workspace": _workspace(num_tokens, topk, device),
            "padded": _padded(num_tokens, row_width),
        }[output]
        launches.clear()
        F.combine_topk_swa_indices(*args, out=out, implementation="triton")
        assert launches == [expected], (len(query_lens), topk, output, launches)


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.bitwise
@pytest.mark.parametrize("owned", [True, False], ids=["owned", "workspace"])
def test_combine_topk_swa_indices_repeatable(accuracy_backend, owned):
    implementation, _, device = accuracy_backend
    args = _inputs([1024, 1024], [4096, 2048], 2, 2048, 2048, 2048, device)
    workspace = _workspace(2048, 2048, device)
    assert_repeatable(
        partial(
            F.combine_topk_swa_indices,
            window_size=WINDOW,
            compress_ratio=2,
            topk=2048,
            M=args[7],
            N=args[8],
            out=None if owned else workspace,
            implementation=implementation,
        ),
        args[:4],
        backward=False,
    )


@pytest.mark.api("functions.combine_topk_swa_indices")
def test_combine_topk_swa_indices_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("combine_topk_swa_indices", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    indices = torch.empty(64, _row_width(TOPK), dtype=torch.int32, device=device)
    lens = torch.empty(64, dtype=torch.int32, device=device)
    torch.library.opcheck(
        namespace.combine_topk_swa_indices_fwd.default,
        (
            torch.zeros(64, WIDTH, dtype=torch.int32, device=device),
            torch.tensor([0, 64], dtype=torch.int32, device=device),
            torch.tensor([64], dtype=torch.int32, device=device),
            torch.tensor([64], dtype=torch.int32, device=device),
            WINDOW,
            2,
            TOPK,
            8192,
            4096,
            indices,
            lens,
            False,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("owned", [True, False], ids=["owned", "workspace"])
def test_combine_topk_swa_indices_compiled(accuracy_backend, backend, owned):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("combine_topk_swa_indices", implementation=implementation)
    args = _inputs([2048], [2048], 2, 2048, 2048, 4096, device)
    topk_indices, query_start_loc, seq_lens, gather_lens = args[:4]
    window_size, compress_ratio, topk, m, n = args[4:]

    def run(candidates, starts, lens, gathers, indices, row_lens):
        return F.combine_topk_swa_indices(
            candidates, starts, lens, gathers, window_size, compress_ratio, topk, m, n,
            out=None if owned else (indices, row_lens),
            implementation=implementation,
        )

    try:
        workspace = _workspace(2048, 2048, device)
        eager = run(topk_indices, query_start_loc, seq_lens, gather_lens, *workspace)
        workspace = _workspace(2048, 2048, device)
        actual = compile_fullgraph(run, backend=backend)(
            topk_indices, query_start_loc, seq_lens, gather_lens, *workspace
        )
        assert actual[0].is_contiguous() and actual[1].is_contiguous()
        assert torch.equal(actual[0], eager[0]) and torch.equal(actual[1], eager[1])
        reference = _workspace(2048, 2048, device) if not owned else None
        expected = F.combine_topk_swa_indices(
            *args, out=reference, implementation="torch"
        )
        assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    finally:
        torch._dynamo.reset()


@pytest.mark.reference
def test_reference_matches_oracle():
    """The torch reference agrees with a per-row oracle written from the definition (CPU)."""
    device = torch.device("cpu")
    torch.manual_seed(11)
    cases = [
        ([5, 0, 7, 3], [9, 4, 40, 10], 2, 6, 6, 20, [4, 0, 12, 10]),
        ([12], [12], 0, 0, 4, 0, None),
        ([6, 6], [20, 6], 3, 5, 8, 7, [6, 6]),
        ([7], [3], 1, 4, 4, 5, None),
    ]
    for query_lens, seq_lens, compress_ratio, topk, width, n_candidates, gather in cases:
        args = _inputs(query_lens, seq_lens, compress_ratio, topk, width, n_candidates, device, seed=3,
                       base=5, gather=gather)
        args = list(args)
        args[0][:, ::3] = -1
        args[0][:, 1::4] = n_candidates + 2
        row_width = _row_width(topk)
        expected_indices = torch.full((sum(query_lens), row_width), -1, dtype=torch.int32)
        expected_lens = torch.zeros(sum(query_lens), dtype=torch.int32)
        written = torch.zeros_like(expected_indices, dtype=torch.bool)
        m, n = args[7], args[8]
        token = 0
        for request, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
            gather_len = args[3][request]
            for offset in range(query_len):
                position = seq_len - query_len + offset
                topk_len = max(min((position + 1) // compress_ratio, topk), 0) if compress_ratio else 0
                for column in range(topk_len):
                    candidate = int(args[0][token, column])
                    expected_indices[token, column] = candidate + m * request if 0 <= candidate < n else -1
                    written[token, column] = True
                swa_start = max(position - (WINDOW - 1), 0, seq_len - gather_len)
                swa_len = max(position - swa_start + 1, 0)
                for column in range(swa_len):
                    expected_indices[token, topk_len + column] = (
                        m * request + n + swa_start + column - (seq_len - gather_len)
                    )
                    written[token, topk_len + column] = True
                expected_lens[token] = topk_len + swa_len
                token += 1
        for owned in (True, False):
            out = None
            if not owned:
                out = (torch.full_like(expected_indices, GARBAGE), torch.full_like(expected_lens, GARBAGE))
            out = F.combine_topk_swa_indices(*args, out=out, implementation="torch")
            if owned:
                assert torch.equal(out[0], expected_indices), (query_lens, compress_ratio, owned)
            else:
                # A workspace keeps its content outside the written entries.
                assert torch.equal(out[0][written], expected_indices[written]), (query_lens, compress_ratio)
                assert (out[0][~written] == GARBAGE).all()
            assert torch.equal(out[1], expected_lens)


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.accuracy
def test_combine_topk_swa_indices_dispatch_paths(accuracy_backend):
    """Both A5 kernels produce the same rows for the shape they share."""
    implementation, _, device = accuracy_backend
    # 256 requests x 8 rows: the row-block kernel takes it on A5, the upstream
    # mapping on A2; both must match the reference.
    args = _inputs([8] * 256, [4096] * 256, 2, 2048, 2048, 2048, device)
    _run(implementation, args, _workspace(2048, 2048, device))
    # One row short of the regime boundary stays on the upstream mapping.
    args = _inputs([9] * 256, [4096] * 256, 2, 2048, 2048, 2048, device)
    _run(implementation, args, _workspace(2304, 2048, device))
