"""Tests for functions.indexer_k_norm_rope_store.

Tolerances follow docs/operator_accuracy_standard.md §2.2: the FP8 values are
compared as 2^-3. The FP32 reduction behind the RMSNorm and the RoPE products
are ordered differently on the two sides, so a value next to an FP8 rounding
boundary can land one code away. Every byte the op does not own (other slots,
skipped tokens, the page padding) is integer data and is compared exactly.
The scale bytes are compared exactly as well; a token whose scale bytes
differ is compared by its dequantized values when both scales are finite
powers of two (``_check``).
"""

import math
import struct

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_bitwise_equal
from tests._checks import assert_close
from tests._compile import compile_fullgraph

HEAD_SIZE = 128
ROPE_DIM = 64
RECORD_BYTES = HEAD_SIZE + 4
MAX_POS = 4096
EPS = 1e-20
FP8_DTYPE = torch.float8_e4m3fn
FP8_TOL = (2.0**-3, 2.0**-3)
# Rows past num_tokens in k_pre and positions hold NaN and a position outside the table.
PAD_ROWS = 3

# Token counts on both sides of the tile threshold (48) and of the tile switch
# (512), tile tails of the 8-row tile (103 % 8, 257 % 8, 300 % 8, 511 % 8) and
# of the 16-row tile (513 % 16, 527 % 16), and production sizes.
SHAPE_CASES = [
    ("T1", 1),
    ("T17", 17),
    ("T47", 47),
    ("T48", 48),
    ("T103", 103),
    ("T257", 257),
    ("T300", 300),
    ("T511", 511),
    ("T512", 512),
    ("T513", 513),
    ("T527", 527),
    ("T2048", 2048),
    ("T8192", 8192),
]


# ------------------------------- case builder ------------------------------ #


def _case(num_tokens, compress_ratio, scale_fmt="fp32", block_size=64, page_pad=0, index_dtype=torch.int64,
          table_dtype=torch.float32, weight_dtype=torch.bfloat16, seed=0, kind="scatter", slots=None, pages=None,
          random_backing=True):
    """CPU inputs: the op reads the first ``num_tokens`` rows of ``k_pre``."""
    g = torch.Generator().manual_seed(seed)
    k_pre = torch.randn(num_tokens + PAD_ROWS, HEAD_SIZE, generator=g).to(torch.bfloat16)
    weight = torch.randn(HEAD_SIZE, generator=g).to(weight_dtype)
    positions = torch.randint(0, MAX_POS, (num_tokens + PAD_ROWS,), generator=g, dtype=index_dtype)
    angles = torch.randn(MAX_POS, ROPE_DIM // 2, generator=g)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(table_dtype)
    pages = 2 * num_tokens // block_size + 2 if pages is None else pages
    page_elems = block_size * RECORD_BYTES + page_pad
    if slots is None:
        slots = torch.randperm(pages * block_size, generator=g)[:num_tokens].to(index_dtype)
        if kind in ("scatter", "garbage") and num_tokens >= 7:
            slots[5::7] = -1
        elif kind == "all_skip":
            slots.fill_(-1)
        elif kind == "run":
            # Slots follow the page table linearly: consecutive inside a page,
            # pages in a random physical order.
            need = num_tokens // block_size + 1
            page_of = torch.randperm(pages, generator=g)[:need]
            slots = (page_of[torch.arange(num_tokens) // block_size] * block_size
                     + torch.arange(num_tokens) % block_size).to(index_dtype)
            slots[5::37] = -1
    if kind == "non_tail":
        positions[:num_tokens] = (positions[:num_tokens] // compress_ratio) * compress_ratio
    if num_tokens == 1:
        # A single-token batch is a group tail.
        positions[0] = compress_ratio - 1
    backing = torch.randint(0, 256, (pages * page_elems,), generator=g, dtype=torch.uint8) if random_backing else None
    if kind == "garbage":
        # Skipped tokens hold NaN and a position outside the table.
        skipped = (slots[:num_tokens] < 0) | ((positions[:num_tokens] + 1) % compress_ratio != 0)
        assert 0 < int(skipped.sum()) < num_tokens
        k_pre[:num_tokens][skipped] = float("nan")
        positions[:num_tokens][skipped] = MAX_POS + 10
    positions[num_tokens:] = MAX_POS + 10
    k_pre[num_tokens:] = float("nan")
    return dict(
        num_tokens=num_tokens, compress_ratio=compress_ratio, scale_fmt=scale_fmt, block_size=block_size,
        pages=pages, page_elems=page_elems, k_pre=k_pre, positions=positions, cos_sin=cos_sin, weight=weight,
        slots=slots, backing=backing,
    )


def _cache_view(case, backing):
    pages, page_elems, block_size = case["pages"], case["page_elems"], case["block_size"]
    return backing.view(pages, page_elems).as_strided((pages, block_size, RECORD_BYTES), (page_elems, RECORD_BYTES, 1))


def _active(case, device):
    """Token ids that write, from the definition."""
    num_tokens = case["num_tokens"]
    positions = case["positions"][:num_tokens].to(device)
    slots = case["slots"].to(device)
    return ((slots >= 0) & ((positions + 1) % case["compress_ratio"] == 0)).nonzero().flatten()


def _cells(case, index, device):
    """Byte offsets of the value cells and of the scale cells of the tokens ``index``."""
    slots = case["slots"].to(device)[index].long()
    page_base = (slots // case["block_size"]) * case["page_elems"]
    row = slots % case["block_size"]
    value = page_base[:, None] + row[:, None] * HEAD_SIZE + torch.arange(HEAD_SIZE, device=device)
    scale = (
        page_base[:, None] + case["block_size"] * HEAD_SIZE + row[:, None] * 4 + torch.arange(4, device=device)
    )
    return value, scale


def _skip_declared(target, implementation, call):
    """Run ``call``; NotImplementedError is the A2 provider's declaration."""
    try:
        return call()
    except NotImplementedError as error:
        assert target.startswith("npu.a2") and implementation != "torch", error
        pytest.skip(f"declared unsupported: {error}")


def _call(implementation, target, case, backing, device, scale_fmt=None):
    def invoke():
        return F.indexer_k_norm_rope_store(
            case["k_pre"].to(device),
            case["positions"].to(device),
            case["cos_sin"].to(device),
            case["weight"].to(device),
            EPS,
            _cache_view(case, backing),
            case["slots"].to(device),
            case["compress_ratio"],
            scale_fmt=case["scale_fmt"] if scale_fmt is None else scale_fmt,
            implementation=implementation,
        )

    return _skip_declared(target, implementation, invoke)


def _scale_words(scale_bytes, scale_fmt):
    """The scale of each token as float32: an FP32 word, or the UE8M0 byte of four equal bytes."""
    if scale_fmt == "mxfp8":
        return torch.exp2(scale_bytes[:, 0].to(torch.float32) - 127)
    return scale_bytes.contiguous().view(torch.float32)[:, 0]


def _is_pow2_scale(scale_bytes, scale_fmt):
    """Tokens whose scale is a finite power of two."""
    if scale_fmt == "mxfp8":
        return ((scale_bytes == scale_bytes[:, :1]).all(dim=-1)) & (scale_bytes[:, 0] != 0xFF)
    bits = scale_bytes.contiguous().view(torch.int32)[:, 0]
    exponent = (bits >> 23) & 0xFF
    return ((bits & 0x7FFFFF) == 0) & (exponent != 0) & (exponent != 0xFF)


def _check(actual, expected, case, device):
    """Values within the FP8 tolerance; scales and all other bytes exact.

    A token whose scale bytes differ passes when both scales are finite powers
    of two and its dequantized values match. An RMSNorm output next to a bf16
    rounding midpoint rounds to the side the order of the FP32 square sums
    favors; when it is the token's absmax on a scale boundary, the scale is the
    next power of two. FP8 is a float format, so a power-of-two scale ratio
    shifts only the exponent and the dequantized values are the same: the codes
    are compared after rescaling to the reference scale.
    """
    index = _active(case, device)
    value, scale = _cells(case, index, device)
    actual, expected = actual.view(-1), expected.view(-1)
    actual_values = actual[value].view(FP8_DTYPE).float()
    expected_values = expected[value].view(FP8_DTYPE).float()
    actual_scale, expected_scale = actual[scale], expected[scale]
    same_scale = (actual_scale == expected_scale).all(dim=-1)
    assert_close(
        actual_values[same_scale], expected_values[same_scale], torch.float32,
        rtol=FP8_TOL[0], atol=FP8_TOL[1], name="values",
    )
    other = ~same_scale
    if bool(other.any()):
        # The rescaling decodes the scale bytes with CPU dtype views.
        actual_scale, expected_scale = actual_scale[other].cpu(), expected_scale[other].cpu()
        fmt = case["scale_fmt"]
        pow2 = _is_pow2_scale(actual_scale, fmt) & _is_pow2_scale(expected_scale, fmt)
        assert bool(pow2.all()), f"scale bytes differ on {int((~pow2).sum())} tokens that are not powers of two"
        ratio = (_scale_words(actual_scale, fmt) / _scale_words(expected_scale, fmt))[:, None]
        assert_close(
            actual_values[other].cpu() * ratio, expected_values[other].cpu(), torch.float32,
            rtol=FP8_TOL[0], atol=FP8_TOL[1], name="values with a differing scale",
        )
    differing = actual != expected
    differing[value] = False
    differing[scale] = False
    assert not bool(differing.any()), "bytes outside the written cells differ"


def _run(implementation, target, case, device, expect_written=True):
    """Run the candidate and the torch reference from equal starting buffers."""
    actual = case["backing"].to(device)
    expected = case["backing"].to(device)
    returned = _call(implementation, target, case, actual, device)
    _call("torch", target, case, expected, device)
    # The cache is written in place and returned as is.
    assert returned.data_ptr() == actual.data_ptr()
    if expect_written:
        assert _active(case, device).numel() > 0
    _check(actual, expected, case, device)
    return actual


# ---------------------------------- tests ---------------------------------- #


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("compress_ratio", [1, 2], ids=["cr1", "cr2"])
@pytest.mark.parametrize("num_tokens", [case[1] for case in SHAPE_CASES], ids=[case[0] for case in SHAPE_CASES])
def test_indexer_k_norm_rope_store_shapes(accuracy_backend, num_tokens, compress_ratio):
    implementation, target, device = accuracy_backend
    _run(implementation, target, _case(num_tokens, compress_ratio, seed=num_tokens), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [1, 17, 2048], ids=["T1", "T17", "T2048"])
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
def test_indexer_k_norm_rope_store_scale_formats(accuracy_backend, num_tokens, scale_fmt):
    implementation, target, device = accuracy_backend
    _run(implementation, target, _case(num_tokens, 2, scale_fmt=scale_fmt, seed=3), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("page_pad", [0, 512], ids=["tight", "padded"])
@pytest.mark.parametrize("block_size", [1, 8, 64, 128], ids=["block1", "block8", "block64", "block128"])
def test_indexer_k_norm_rope_store_page_geometry(accuracy_backend, block_size, page_pad):
    implementation, target, device = accuracy_backend
    _run(implementation, target, _case(300, 2, block_size=block_size, page_pad=page_pad, seed=block_size), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 300], ids=["T17", "T300"])
def test_indexer_k_norm_rope_store_contiguous_slots(accuracy_backend, num_tokens):
    """Slots follow the page table: consecutive inside a page, pages out of order."""
    implementation, target, device = accuracy_backend
    _run(implementation, target, _case(num_tokens, 1, kind="run", block_size=64, page_pad=128, seed=5), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 300], ids=["T17", "T300"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_indexer_k_norm_rope_store_index_dtypes(accuracy_backend, num_tokens, index_dtype):
    implementation, target, device = accuracy_backend
    _run(implementation, target, _case(num_tokens, 2, index_dtype=index_dtype, seed=7), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16], ids=["cos_sin_fp32", "cos_sin_bf16"])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["weight_bf16", "weight_fp32"])
def test_indexer_k_norm_rope_store_table_dtypes(accuracy_backend, table_dtype, weight_dtype):
    implementation, target, device = accuracy_backend
    for num_tokens in (17, 300):
        _run(implementation, target, _case(num_tokens, 2, table_dtype=table_dtype, weight_dtype=weight_dtype,
                                           seed=11), device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 300], ids=["T17", "T300"])
def test_indexer_k_norm_rope_store_input_layouts(accuracy_backend, num_tokens):
    """A row pitch above 128 in k_pre and above 64 in cos_sin_cache is honoured."""
    implementation, target, device = accuracy_backend
    case = _case(num_tokens, 2, seed=13)
    k_pre = torch.zeros(num_tokens + PAD_ROWS, 256, dtype=torch.bfloat16)
    k_pre[:, :HEAD_SIZE] = case["k_pre"]
    case["k_pre"] = k_pre[:, :HEAD_SIZE]
    table = torch.zeros(MAX_POS + 8, 128, dtype=case["cos_sin"].dtype)
    table[:MAX_POS, :ROPE_DIM] = case["cos_sin"]
    case["cos_sin"] = table[:, :ROPE_DIM]
    assert case["k_pre"].stride(0) == 256 and case["cos_sin"].stride(0) == 128
    _run(implementation, target, case, device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("kind", ["all_skip", "non_tail", "garbage"])
def test_indexer_k_norm_rope_store_skipped_tokens(accuracy_backend, kind):
    """Tokens without a slot or off the group tail are not read and write nothing."""
    implementation, target, device = accuracy_backend
    for num_tokens in (17, 300):
        case = _case(num_tokens, 2, kind=kind, block_size=64, page_pad=128, seed=17 + num_tokens)
        untouched = kind in ("all_skip", "non_tail")
        actual = _run(implementation, target, case, device, expect_written=not untouched)
        if untouched:
            assert torch.equal(actual, case["backing"].to(device))


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 300], ids=["T17", "T300"])
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
def test_indexer_k_norm_rope_store_zero_rows(accuracy_backend, num_tokens, scale_fmt):
    """A zero row takes the scale floor: scale 2**-22 and zero values."""
    implementation, target, device = accuracy_backend
    case = _case(num_tokens, 1, scale_fmt=scale_fmt, seed=19)
    case["k_pre"][:num_tokens].zero_()
    actual = _run(implementation, target, case, device)
    value, scale = _cells(case, _active(case, device), device)
    if scale_fmt == "fp32":
        expected_scale = torch.tensor(list(struct.pack("<f", 2.0**-22)), dtype=torch.uint8, device=device)
    else:
        expected_scale = torch.full((4,), 127 - 22, dtype=torch.uint8, device=device)
    assert torch.equal(actual.view(-1)[scale], expected_scale.expand(scale.shape))
    assert bool((actual.view(-1)[value] & 0x7F == 0).all())


def _boundary_rows(count, weight, seed):
    """Rows whose largest value lies below column 64 (the rotation keeps a pair's
    norm, so the tail is bounded by its largest pair norm) and rounds to
    ``224 * 2**k`` in bf16 from within a fifth of a bf16 step, so that the FP32
    rounding differences between the two sides cannot move it off that code.
    Returns ``(rows, columns, k)``."""
    g = torch.Generator().manual_seed(seed)
    rows, columns, exponents = [], [], []
    while len(rows) < count:
        k = torch.randn(4096, HEAD_SIZE, generator=g).to(torch.bfloat16)
        normed = k.double() * torch.rsqrt(k.double().square().mean(-1, keepdim=True) + EPS) * weight.double()
        head_value, column = normed[:, :ROPE_DIM].abs().max(-1)
        # A rotated pair keeps its norm, so the tail can reach the largest pair norm.
        even, odd = normed[:, ROPE_DIM::2], normed[:, ROPE_DIM + 1 ::2]
        tail_value = (even.square() + odd.square()).sqrt().amax(-1)
        mantissa = head_value / torch.exp2(head_value.log2().floor())
        grid = (mantissa - 1) * 128
        # Just above the 224 code, or just below it (in the last fifth of the 223 bin).
        on_boundary = ((grid.floor() == 224 - 128) & (grid - 96 < 0.2)) | (
            (grid.floor() == 223 - 128) & (grid - 95 > 0.8)
        )
        keep = on_boundary & (tail_value < 0.97 * head_value)
        for i in keep.nonzero().flatten().tolist():
            rows.append(k[i])
            columns.append(int(column[i]))
            # absmax = 224 * 2**k, so the scale is 2**(k - 1).
            exponents.append(int(head_value[i].log2().floor()) - 7)
    return torch.stack(rows[:count]), torch.tensor(columns[:count]), exponents[:count]


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 96], ids=["T17", "T96"])
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
def test_indexer_k_norm_rope_store_scale_boundary(accuracy_backend, num_tokens, scale_fmt):
    """absmax exactly ``224 * 2**k`` (the ratio to 448 is a power of two): the scale is
    the lower power of two and the largest value quantizes to 448, for every scale
    exponent from the floor up to 12."""
    implementation, target, device = accuracy_backend
    # Every row writes: no skipped slot may shift the row pairing.
    case = _case(num_tokens, 1, scale_fmt=scale_fmt, seed=23, kind="solid")
    rows, columns, exponents = _boundary_rows(num_tokens, case["weight"], seed=29)
    case["k_pre"][:num_tokens] = rows
    base_weight = case["weight"].double()
    exponents = torch.tensor(exponents, device=device)
    columns = columns.to(device)
    # Scale exponent of row r after the weight is scaled by 2**s: exponents[r] + s - 1.
    # s keeps every row inside [-22, 12].
    for s in range(-21 - int(exponents.min()), 14 - int(exponents.max())):
        case["weight"] = (base_weight * 2.0**s).to(torch.bfloat16)
        actual = _run(implementation, target, case, device)
        index = _active(case, device)
        value, scale = _cells(case, index, device)
        scale_exponent = exponents + s - 1
        if scale_fmt == "fp32":
            expected_scale = torch.tensor(
                [list(struct.pack("<f", 2.0**e)) for e in scale_exponent.cpu().tolist()], dtype=torch.uint8
            ).to(device)
        else:
            expected_scale = (127 + scale_exponent).to(torch.uint8)[:, None].expand(-1, 4)
        assert torch.equal(actual.view(-1)[scale], expected_scale), f"scale at weight scale 2**{s}"
        # absmax / scale = 448: the fp8 code 0x7E with the sign of the value.
        largest = actual.view(-1)[value].view(num_tokens, HEAD_SIZE)[torch.arange(num_tokens, device=device), columns]
        assert bool((largest & 0x7F == 0x7E).all()), f"largest value at weight scale 2**{s}"


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
def test_indexer_k_norm_rope_store_check_differing_scale(accuracy_backend, scale_fmt):
    """_check accepts halved codes at a doubled power-of-two scale, and still fails a
    zeroed code, a non-power-of-two scale, and a zeroed code at an unchanged scale."""
    implementation, target, device = accuracy_backend
    case = _case(300, 2, scale_fmt=scale_fmt, seed=31)
    actual = case["backing"].to(device)
    expected = case["backing"].to(device)
    _call(implementation, target, case, actual, device)
    _call("torch", target, case, expected, device)
    _check(actual, expected, case, device)
    # Requantize every third written token at twice the scale: halved codes and a
    # doubled power-of-two scale dequantize to the same values.
    written = _active(case, device)
    index = written[1::3]
    value, scale = _cells(case, index, device)
    flat = actual.view(-1)
    flat[value] = (flat[value].view(FP8_DTYPE).float() / 2.0).to(FP8_DTYPE).view(torch.uint8)
    if scale_fmt == "fp32":
        words = _scale_words(flat[scale].cpu(), scale_fmt) * 2.0
        flat[scale] = words[:, None].contiguous().view(torch.uint8).to(device)
    else:
        flat[scale] += 1
    _check(actual, expected, case, device)
    # A requantized code of magnitude >= 64 zeroed out must fail the rescaled comparison.
    requant = flat[value].view(FP8_DTYPE).float()
    token, column = (requant.abs() >= 64).nonzero()[0].tolist()
    broken = actual.clone()
    broken.view(-1)[value[token, column]] = 0
    with pytest.raises(AssertionError):
        _check(broken, expected, case, device)
    # A non-power-of-two scale must fail the power-of-two gate.
    broken = actual.clone()
    scale_one = _cells(case, index[:1], device)[1]
    if scale_fmt == "fp32":
        words = _scale_words(broken.view(-1)[scale_one].cpu(), scale_fmt) * 1.5
        broken.view(-1)[scale_one] = words[:, None].contiguous().view(torch.uint8).to(device)
    else:
        broken.view(-1)[scale_one[:, :1]] += 1
    with pytest.raises(AssertionError, match="not powers of two"):
        _check(broken, expected, case, device)
    # A code of magnitude >= 64 zeroed out at an unchanged scale must fail the exact comparison.
    broken = actual.clone()
    value_one = _cells(case, written[:1], device)[0]
    codes = broken.view(-1)[value_one].view(FP8_DTYPE).float()
    column = (codes[0].abs() >= 64).nonzero()[0, 0].item()
    broken.view(-1)[value_one[0, column]] = 0
    with pytest.raises(AssertionError):
        _check(broken, expected, case, device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [17, 300, 2048], ids=["T17", "T300", "T2048"])
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
def test_indexer_k_norm_rope_store_special_values(accuracy_backend, num_tokens, scale_fmt):
    """NaN and +-inf in written rows and in the weight: NaN occupies the same cells,
    the scales agree and the other cells follow the FP8 tolerance, on the per-token
    kernel and on both tiles of the tile kernel."""
    implementation, target, device = accuracy_backend
    case = _case(num_tokens, 1, scale_fmt=scale_fmt, seed=31)
    for token, column, value in [
        (0, 0, float("nan")),
        (1, 5, float("nan")),
        (2, 9, float("inf")),
        (3, 70, float("-inf")),
        (4, 127, float("inf")),
        (6, 5, float("nan")),  # NaN and inf in one row
        (6, 90, float("inf")),
        (num_tokens - 2, 100, float("nan")),  # the last tile
    ]:
        case["k_pre"][token, column] = value
    weight = case["weight"]
    # An inf weight on a rope column gives every row an infinite absmax; on a
    # norm column only that cell is non-finite, as in the reference.
    for column, value in [(None, None), (71, float("inf")), (3, float("inf")), (80, float("nan"))]:
        case["weight"] = weight.clone()
        if column is not None:
            case["weight"][column] = value
        actual = case["backing"].to(device)
        expected = case["backing"].to(device)
        _call(implementation, target, case, actual, device)
        _call("torch", target, case, expected, device)
        index = _active(case, device)
        value, scale = _cells(case, index, device)
        actual, expected = actual.view(-1), expected.view(-1)
        nan = torch.isnan(actual[value].view(FP8_DTYPE).float())
        assert torch.equal(nan, torch.isnan(expected[value].view(FP8_DTYPE).float())), column
        assert_close(
            actual[value][~nan].view(FP8_DTYPE).float(), expected[value][~nan].view(FP8_DTYPE).float(),
            torch.float32, rtol=FP8_TOL[0], atol=FP8_TOL[1], name="values",
        )
        if scale_fmt == "mxfp8":
            assert torch.equal(actual[scale], expected[scale]), column
        else:
            actual_scale = actual[scale].view(torch.float32)
            expected_scale = expected[scale].view(torch.float32)
            finite = ~torch.isnan(actual_scale) & ~torch.isnan(expected_scale)
            assert torch.equal(torch.isnan(actual_scale), torch.isnan(expected_scale)), column
            assert torch.equal(actual_scale[finite], expected_scale[finite]), column


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_indexer_k_norm_rope_store_large_cache(accuracy_backend, index_dtype):
    """A cache of 2**31 bytes or more: the slot in the last page is written exactly."""
    implementation, target, device = accuracy_backend
    block_size, page_elems = 64, 64 * RECORD_BYTES
    pages = 2**31 // page_elems + 3
    last_page = [(pages - 1) * block_size + i for i in (0, 5, 63)]
    slots = torch.tensor([3, 65, block_size - 1] + last_page, dtype=index_dtype)
    case = _case(6, 1, index_dtype=index_dtype, block_size=block_size, slots=slots, pages=pages, seed=37,
                 random_backing=False)
    assert case["page_elems"] * pages >= 2**31
    actual = torch.randint(0, 256, (pages * page_elems,), dtype=torch.uint8, device=device)
    expected = actual.clone()
    _call(implementation, target, case, actual, device)
    _call("torch", target, case, expected, device)
    _check(actual, expected, case, device)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
def test_indexer_k_norm_rope_store_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): no launch, the cache is untouched."""
    implementation, target, device = accuracy_backend
    case = _case(0, 2, seed=41)
    backing = case["backing"].to(device)
    before = backing.clone()
    returned = F.indexer_k_norm_rope_store(
        case["k_pre"].to(device), case["positions"].to(device)[:0], case["cos_sin"].to(device),
        case["weight"].to(device), EPS, _cache_view(case, backing), case["slots"].to(device), 2,
        implementation=implementation,
    )
    assert case["slots"].numel() == 0 and case["k_pre"].shape == (PAD_ROWS, HEAD_SIZE)
    assert returned.data_ptr() == backing.data_ptr()
    assert torch.equal(backing, before)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
def test_indexer_k_norm_rope_store_fp4_cache_unsupported(accuracy_backend):
    implementation, _, device = accuracy_backend
    case = _case(17, 2, seed=43)
    with pytest.raises(NotImplementedError, match="MXFP4"):
        F.indexer_k_norm_rope_store(
            case["k_pre"].to(device), case["positions"].to(device), case["cos_sin"].to(device),
            case["weight"].to(device), EPS, _cache_view(case, case["backing"].to(device)),
            case["slots"].to(device), 2, use_fp4_cache=True, implementation=implementation,
        )


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
def test_indexer_k_norm_rope_store_a2_declares_unsupported(accuracy_backend):
    """The A2 provider raises before any launch; the contract checks still run there."""
    implementation, target, device = accuracy_backend
    if not (target.startswith("npu.a2") and implementation in (None, "triton")):
        pytest.skip("A2 declaration")
    case = _case(17, 2, seed=47)
    with pytest.raises(NotImplementedError, match="910B"):
        F.indexer_k_norm_rope_store(
            case["k_pre"].to(device), case["positions"].to(device), case["cos_sin"].to(device),
            case["weight"].to(device), EPS, _cache_view(case, case["backing"].to(device)),
            case["slots"].to(device), 2, implementation=implementation,
        )


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "broken,match",
    [
        ("scale_fmt", "scale_fmt must be one of"),
        ("k_pre_dtype", "k_pre must be a bf16"),
        ("k_pre_width", "k_pre must be a bf16"),
        ("k_pre_stride", "k_pre must have unit column stride"),
        ("positions_dtype", "positions must be a 1D int32/int64"),
        ("positions_ndim", "positions must be a 1D int32/int64"),
        ("positions_device", "must be on the k_pre device"),
        ("slots_dtype", "kv_slot_mapping must be a 1D int32/int64"),
        ("slots_length", "kv_slot_mapping cannot be longer"),
        ("cos_sin_shape", "cos_sin_cache must be an \\[max_pos, 64\\]"),
        ("cos_sin_dtype", "cos_sin_cache must be an fp32/bf16"),
        ("cos_sin_stride", "cos_sin_cache must have unit column stride"),
        ("weight_shape", "rms_norm_weight must be a \\[128\\]"),
        ("weight_dtype", "rms_norm_weight must be a bf16/fp32"),
        ("weight_stride", "rms_norm_weight must be contiguous"),
        ("compress_ratio", "compress_ratio must be 1 or 2"),
        ("cache_ndim", "k_cache must be a uint8 \\[pages, block_size, 132\\]"),
        ("cache_dtype", "k_cache must be a uint8 \\[pages, block_size, 132\\]"),
        ("cache_record_stride", "unit record stride and a 132-byte row stride"),
        ("cache_page_small", "page stride must be a multiple of 4 and at least"),
        ("cache_page_unaligned", "page stride must be a multiple of 4 and at least"),
    ],
)
def test_indexer_k_norm_rope_store_rejects_invalid_inputs(accuracy_backend, broken, match):
    """Each contract check raises before any kernel is launched."""
    implementation, _, device = accuracy_backend
    case = _case(64, 2, seed=53)
    pages, page_elems, block_size = case["pages"], case["page_elems"], case["block_size"]
    # Broken layouts are built on the device: a copy would turn them contiguous.
    k_pre = case["k_pre"].to(device)
    cos_sin = case["cos_sin"].to(device)
    weight = case["weight"].to(device)
    backing = case["backing"].to(device)
    args = dict(
        k_pre=k_pre, positions=case["positions"].to(device), cos_sin_cache=cos_sin, rms_norm_weight=weight,
        rms_norm_eps=EPS, k_cache=_cache_view(case, backing), kv_slot_mapping=case["slots"].to(device),
        compress_ratio=2, scale_fmt="fp32",
    )
    if broken == "scale_fmt":
        args["scale_fmt"] = "ue8m0"
    elif broken == "k_pre_dtype":
        args["k_pre"] = k_pre.float()
    elif broken == "k_pre_width":
        args["k_pre"] = k_pre[:, :64].contiguous()
    elif broken == "k_pre_stride":
        wide = torch.zeros(k_pre.shape[0], 2 * HEAD_SIZE, dtype=torch.bfloat16, device=device)
        wide[:, ::2] = k_pre
        args["k_pre"] = wide[:, ::2]
    elif broken == "positions_dtype":
        args["positions"] = args["positions"].float()
    elif broken == "positions_ndim":
        args["positions"] = args["positions"].unsqueeze(0)
    elif broken == "positions_device":
        args["positions"] = args["positions"].cpu()
    elif broken == "slots_dtype":
        args["kv_slot_mapping"] = args["kv_slot_mapping"].to(torch.int16)
    elif broken == "slots_length":
        args["kv_slot_mapping"] = torch.zeros(k_pre.shape[0] + 1, dtype=torch.int64, device=device)
    elif broken == "cos_sin_shape":
        args["cos_sin_cache"] = cos_sin[:, :32].contiguous()
    elif broken == "cos_sin_dtype":
        args["cos_sin_cache"] = cos_sin.to(torch.float16)
    elif broken == "cos_sin_stride":
        wide = torch.zeros(MAX_POS, 2 * ROPE_DIM, dtype=torch.float32, device=device)
        wide[:, ::2] = cos_sin
        args["cos_sin_cache"] = wide[:, ::2]
    elif broken == "weight_shape":
        args["rms_norm_weight"] = weight[:64]
    elif broken == "weight_dtype":
        args["rms_norm_weight"] = weight.to(torch.float16)
    elif broken == "weight_stride":
        wide = torch.zeros(2 * HEAD_SIZE, dtype=torch.bfloat16, device=device)
        wide[::2] = weight
        args["rms_norm_weight"] = wide[::2]
    elif broken == "compress_ratio":
        args["compress_ratio"] = 3
    elif broken == "cache_ndim":
        args["k_cache"] = args["k_cache"][0]
    elif broken == "cache_dtype":
        args["k_cache"] = args["k_cache"].view(torch.int8)
    elif broken == "cache_record_stride":
        wide = torch.zeros(pages, block_size * (RECORD_BYTES + 4), dtype=torch.uint8, device=device)
        args["k_cache"] = wide.as_strided(
            (pages, block_size, RECORD_BYTES), (block_size * (RECORD_BYTES + 4), RECORD_BYTES + 4, 1)
        )
    elif broken == "cache_page_small":
        args["k_cache"] = backing.view(pages, page_elems).as_strided(
            (pages, block_size, RECORD_BYTES), ((block_size - 1) * RECORD_BYTES, RECORD_BYTES, 1)
        )
    elif broken == "cache_page_unaligned":
        padded = torch.zeros(pages * (page_elems + 2), dtype=torch.uint8, device=device)
        args["k_cache"] = padded.view(pages, page_elems + 2).as_strided(
            (pages, block_size, RECORD_BYTES), (page_elems + 2, RECORD_BYTES, 1)
        )
    with pytest.raises(ValueError, match=match):
        F.indexer_k_norm_rope_store(**args, implementation=implementation)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.bitwise
@pytest.mark.parametrize("num_tokens", [300, 2048], ids=["T300", "T2048"])
def test_indexer_k_norm_rope_store_repeatable(accuracy_backend, num_tokens):
    """Equal starting buffers give equal bytes, and a second identical store changes nothing."""
    implementation, target, device = accuracy_backend
    case = _case(num_tokens, 2, seed=59)
    first = case["backing"].to(device)
    second = case["backing"].to(device)
    _call(implementation, target, case, first, device)
    _call(implementation, target, case, second, device)
    assert_bitwise_equal(first, second)
    _call(implementation, target, case, second, device)
    assert_bitwise_equal(first, second)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
def test_indexer_k_norm_rope_store_dispatch_paths(accuracy_backend, monkeypatch):
    """The token count and the cache size select the kernel and, on the tile path,
    the tile: 8 rows with 4 warps below 512 tokens, 16 rows with 8 warps from
    there on."""
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("A5 Triton dispatch")
    preload("indexer_k_norm_rope_store", implementation="triton", target=target)
    from mojo_opset.kernels.npu_a5_triton import indexer_k_norm_rope_store as provider

    launches = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(
                    (self.name, grid[0], kwargs.get("RPP"), kwargs.get("num_warps"), kwargs.get("compile_mode"))
                )

            return launch

    monkeypatch.setattr(provider, "_indexer_k_norm_rope_store_rows", _Recorder("tile"))
    monkeypatch.setattr(provider, "_indexer_k_norm_rope_quant_store_kernel", _Recorder("token"))
    case = _case(8192, 2, seed=61)
    block_size = case["block_size"]
    assert case["page_elems"] % 4 == 0
    k_pre, positions = case["k_pre"].to(device), case["positions"].to(device)
    cos_sin, weight = case["cos_sin"].to(device), case["weight"].to(device)
    slots = case["slots"].to(device)
    small = _cache_view(case, case["backing"].to(device))
    # 2**31 bytes or more: the page offsets need 64 bits.
    big_pages = 2**31 // case["page_elems"] + 1
    big = torch.empty(big_pages * case["page_elems"], dtype=torch.uint8, device=device)
    assert big.numel() >= 2**31
    big = big.view(big_pages, case["page_elems"]).as_strided(
        (big_pages, block_size, RECORD_BYTES), (case["page_elems"], RECORD_BYTES, 1)
    )
    tiles = {"small": (8, 4), "large": (16, 8)}
    for num_tokens, big_cache, path, tile in [
        (0, False, None, None),
        (1, False, "token", None),
        (47, False, "token", None),
        (48, False, "tile", "small"),
        (103, False, "tile", "small"),
        (511, False, "tile", "small"),
        (512, False, "tile", "large"),
        (527, False, "tile", "large"),
        (8192, False, "tile", "large"),
        (48, True, "token", None),
        (8192, True, "token", None),
    ]:
        launches.clear()
        F.indexer_k_norm_rope_store(
            k_pre, positions, cos_sin, weight, EPS, big if big_cache else small, slots[:num_tokens], 2,
            implementation="triton",
        )
        if path is None:
            expected = []
        elif path == "token":
            expected = [("token", num_tokens, None, 1, None)]
        else:
            rpp, warps = tiles[tile]
            expected = [("tile", -(-num_tokens // rpp), rpp, warps, "simt_only")]
        assert launches == expected, (num_tokens, big_cache, launches)


@pytest.mark.api("functions.indexer_k_norm_rope_store")
def test_indexer_k_norm_rope_store_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    if target.startswith("npu.a2"):
        # opcheck executes the op, and the A2 provider declares it unsupported.
        pytest.skip("declared unsupported on A2")
    preload("indexer_k_norm_rope_store", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    case = _case(8, 2, block_size=8, seed=67)
    torch.library.opcheck(
        namespace.indexer_k_norm_rope_store_fwd.default,
        (
            case["k_pre"].to(device),
            case["positions"].to(device),
            case["cos_sin"].to(device),
            case["weight"].to(device),
            EPS,
            _cache_view(case, case["backing"].to(device)),
            case["slots"].to(device),
            2,
            False,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_indexer_k_norm_rope_store_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("indexer_k_norm_rope_store", implementation=implementation)
    case = _case(300, 2, seed=71)
    k_pre, positions = case["k_pre"].to(device), case["positions"].to(device)
    cos_sin, weight, slots = case["cos_sin"].to(device), case["weight"].to(device), case["slots"].to(device)

    def run(cache, k_pre, positions, cos_sin, weight, slots):
        F.indexer_k_norm_rope_store(
            k_pre, positions, cos_sin, weight, EPS, cache, slots, 2, implementation=implementation
        )
        return (cache,)

    inputs = (k_pre, positions, cos_sin, weight, slots)

    def compiled(backing):
        return compile_fullgraph(run, backend=backend)(_cache_view(case, backing), *inputs)

    try:
        eager_backing = case["backing"].to(device)
        (eager,) = _skip_declared(target, implementation, lambda: run(_cache_view(case, eager_backing), *inputs))
        compiled_backing = case["backing"].to(device)
        (actual,) = _skip_declared(target, implementation, lambda: compiled(compiled_backing))
        assert_bitwise_equal(actual, eager)
    finally:
        torch._dynamo.reset()


# --------------------------- reference-only checks --------------------------- #


def _oracle_backing(case):
    """Per-token transcription of the definition, with the scale exponent from libm."""
    expected = case["backing"].clone()
    compress_ratio, block_size, page_elems = case["compress_ratio"], case["block_size"], case["page_elems"]
    cos_sin = case["cos_sin"].float()
    weight = case["weight"].float()
    for token in range(case["num_tokens"]):
        slot, position = int(case["slots"][token]), int(case["positions"][token])
        if slot < 0 or (position + 1) % compress_ratio:
            continue
        k = case["k_pre"][token].float()
        k = k * torch.rsqrt(k.square().mean() + EPS)
        k = (k * weight).to(torch.bfloat16).float()
        row = cos_sin[position // compress_ratio * compress_ratio]
        even, odd = k[ROPE_DIM::2], k[ROPE_DIM + 1 ::2]
        rotated = torch.stack((even * row[: ROPE_DIM // 2] - odd * row[ROPE_DIM // 2 :],
                               odd * row[: ROPE_DIM // 2] + even * row[ROPE_DIM // 2 :]), dim=-1).flatten()
        k = torch.cat((k[:ROPE_DIM], rotated)).to(torch.bfloat16).float()
        absmax = max(float(k.abs().max()), 1e-4)
        scale = 2.0 ** math.ceil(math.log2(absmax / 448.0))
        values = (k / scale).clamp(-448.0, 448.0).to(FP8_DTYPE).view(torch.uint8)
        if case["scale_fmt"] == "mxfp8":
            scale_bytes = bytes([math.frexp(scale)[1] + 126] * 4)
        else:
            scale_bytes = struct.pack("<f", scale)
        page, row_in_page = slot // block_size, slot % block_size
        base = page * page_elems + row_in_page * HEAD_SIZE
        expected[base:base + HEAD_SIZE] = values
        base = page * page_elems + block_size * HEAD_SIZE + row_in_page * 4
        expected[base:base + 4] = torch.tensor(list(scale_bytes), dtype=torch.uint8)
    return expected


@pytest.mark.reference
@pytest.mark.parametrize("scale_fmt", ["fp32", "mxfp8"])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_reference_matches_oracle(scale_fmt, compress_ratio):
    """The torch provider must equal the per-token transcription on CPU."""
    case = _case(40, compress_ratio, scale_fmt=scale_fmt, block_size=8, page_pad=128, seed=73)
    case["weight"] = case["weight"].float()
    backing = case["backing"].clone()
    F.indexer_k_norm_rope_store(
        case["k_pre"], case["positions"], case["cos_sin"], case["weight"], EPS, _cache_view(case, backing),
        case["slots"], compress_ratio, scale_fmt=scale_fmt, implementation="torch",
    )
    assert torch.equal(backing, _oracle_backing(case))
