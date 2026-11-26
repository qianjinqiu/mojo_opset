"""Indexer K store for DeepSeek V4.1 kv-source layers (A5): k_norm, RoPE, FP8 quant,
paged write.

``k = k_norm(wk(latent))`` at the group-tail tokens, RoPE'd at the group's first
position and quantized into the paged indexer K cache. The ``wk`` projection runs
outside.

Two kernels:

* the per-token kernel of upstream (one program per token, early return for
  skipped tokens), with int64 page offsets. The row is only 128 wide, so every
  program issues the whole instruction sequence for one row and the launch is
  issue-bound: it serves batches below ``_TILE_MIN_TOKENS`` and caches of
  ``2**31`` bytes or more, whose page offsets need 64 bits. It runs in the
  default compile mode.
* a 2D tile kernel from ``_TILE_MIN_TOKENS`` up: 8 rows with 4 warps below
  ``_TILE_LARGE_MIN_TOKENS``, 16 rows with 8 warps from there on, launched
  with ``compile_mode="simt_only"`` because the default mode moves the weight
  vector and the row indexes through scalar element loops. The SIMT lowering
  rounds the fp32 arithmetic differently, so a value next to an FP8 rounding
  boundary can differ by one code from the per-token kernel and from the
  reference. A NaN or inf in ``k_pre`` makes the row all NaN, as in the
  reference. The even and odd columns are loaded as two stride-2 tiles because
  ``tl.split`` on the row-major tile costs more than the strided loads;
  ``tl.join`` runs once, before the store.

A page holds ``block_size * 128`` value bytes followed by ``block_size * 4``
scale bytes. The scale is assembled from its integer exponent: ``exp2`` is not
correctly rounded for integer arguments on this backend.
"""

import torch
import triton
import triton.language as tl

# Below this token count one program per row is faster than a tile.
_TILE_MIN_TOKENS = 48
# From this token count the 16-row tile with 8 warps is faster than the 8-row tile with 4 warps.
_TILE_LARGE_MIN_TOKENS = 512
_TILE_SMALL = dict(RPP=8, num_warps=4)
_TILE_LARGE = dict(RPP=16, num_warps=8)
# Below this cache size every page offset fits in int32.
_INT32_CACHE_BYTES = 2**31
_FP8_MAX = 448.0
# Module-level constants referenced inside a kernel must be constexpr instances.
_SCALE_FLOOR = tl.constexpr(1e-4)


@triton.jit
def _pack_ue8m0_x4(scale):
    """Encode a power-of-two fp32 scale as four equal packed UE8M0 bytes."""
    return ((scale.to(tl.int32, bitcast=True) >> 23) & 0xFF) * 0x01010101


@triton.jit
def _pow2_scale(absmax, eps: tl.constexpr, fp8_max: tl.constexpr):
    """``2 ** ceil(log2(max(absmax, eps) / fp8_max))`` as float32.

    The exponent is read from the bits of the argument: the exponent field is
    ``floor(log2(x))`` and a non-zero mantissa steps up. ``log2`` on this backend
    is one ULP high on a few exact powers of two, where ``ceil`` then returns the
    next exponent. The scale is assembled from the integer exponent,
    ``(e + 127) << 23``: ``tl.math.exp2`` is not correctly rounded for integer
    arguments on this backend. A NaN absmax gives NaN; an infinite one gives
    +inf (all-ones exponent), like the reference.
    """
    x = tl.maximum(absmax, eps, propagate_nan=tl.PropagateNan.ALL) * (1.0 / fp8_max)
    nan = x != x
    bits = x.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return tl.where(nan, float("nan"), ((e + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True))


@triton.jit
def _pow2_recip(scale):
    """1 / scale for a power-of-two, NaN or +inf scale, from the exponent field."""
    bits = scale.to(tl.uint32, bitcast=True)
    inv = ((254 - (bits >> 23)) << 23).to(tl.float32, bitcast=True)
    return tl.where(scale != scale, float("nan"), tl.where(scale == float("inf"), 0.0, inv))


@triton.jit
def _indexer_k_norm_rope_quant_store_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MX_SCALE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    # Only the last token of a group publishes that group's index key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, HEAD_SIZE)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block).to(tl.float32)
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)
    k = k.to(tl.float32)

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))  # each [NUM_PAIRS]
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    # Only the rope pairs are rotated; the other pairs keep their norm values.
    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = tl.where(is_rope_pair, even * cos_v - odd * sin_v, even)
    new_odd = tl.where(is_rope_pair, odd * cos_v + even * sin_v, odd)
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ── Paged cache pointers (segregated: values first, then scales) ──
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    # Per-token FP8 (single 128-wide block) with one float32 scale.
    result = tl.interleave(new_even, new_odd)  # [HEAD_SIZE] fp32
    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=0, propagate_nan=tl.PropagateNan.ALL)
    scale = _pow2_scale(absmax, _SCALE_FLOOR, FP8_MAX)
    x_clamped = tl.clamp(
        result_bf16 * _pow2_recip(scale), -FP8_MAX, FP8_MAX, propagate_nan=tl.PropagateNan.ALL
    )
    x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(val_ptr + block, x_uint8)
    if MX_SCALE:
        tl.store(scale_ptr.to(tl.pointer_type(tl.int32)), _pack_ue8m0_x4(scale))
    else:
        tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale)


@triton.jit
def _indexer_k_norm_rope_store_rows(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    num_tokens,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MX_SCALE: tl.constexpr,
    RPP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)
    live = rows < num_tokens
    kv_slot = tl.load(kv_slot_mapping_ptr + rows, mask=live, other=-1)
    position = tl.load(positions_ptr + rows, mask=live, other=0)
    # Only the last token of a group with a valid slot publishes a key.
    active = live & (kv_slot >= 0) & ((position + 1) % COMPRESS_RATIO == 0)
    mask = active[:, None]
    # int32 for every offset derived from the indexes: both fit in 32 bits for
    # any reachable cache or position, and byte-level cache offsets are kept
    # int32 on this backend.
    kv_slot = kv_slot.to(tl.int32)

    position = position.to(tl.int32)

    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    pair_idx = tl.arange(0, NUM_PAIRS)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    # The even and odd columns are loaded as two stride-2 tiles: GPT-J RoPE
    # pairs column 2i with 2i+1, and tl.split on the row-major tile costs
    # more than the strided loads.
    even = tl.load(
        k_pre_ptr + rows[:, None] * k_pre_stride + 2 * pair_idx[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    odd = tl.load(
        k_pre_ptr + rows[:, None] * k_pre_stride + 2 * pair_idx[None, :] + 1, mask=mask, other=0.0
    ).to(tl.float32)
    w_even = tl.load(rms_norm_weight_ptr + 2 * pair_idx).to(tl.float32)
    w_odd = tl.load(rms_norm_weight_ptr + 2 * pair_idx + 1).to(tl.float32)
    # A NaN or inf in k_pre makes the whole row NaN, as in the reference. In
    # the SIMT lowering such a value does not reach every element of its row
    # through the norm, so the pair is zeroed here and marked with a NaN after
    # the norm. No finite bf16 value reaches 3.4e38, so the comparisons select
    # NaN and inf.
    nonfinite = (even != even) | (tl.abs(even) >= 3.4e38) | (odd != odd) | (tl.abs(odd) >= 3.4e38)
    even = tl.where(nonfinite, 0.0, even)
    odd = tl.where(nonfinite, 0.0, odd)
    variance = (tl.sum(even * even, axis=1) + tl.sum(odd * odd, axis=1)) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    even = (even * rrms[:, None] * w_even[None, :]).to(tl.bfloat16).to(tl.float32)
    odd = (odd * rrms[:, None] * w_odd[None, :]).to(tl.bfloat16).to(tl.float32)
    even = tl.where(nonfinite, float("nan"), even)
    odd = tl.where(nonfinite, float("nan"), odd)

    # ── GPT-J forward RoPE in fp32 ────────────────────────────────────
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio.
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    # Skipped rows load nothing: their position may lie outside the cache.
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cs_mask = mask & is_rope_pair[None, :]
    cache_base = cos_sin_cache_ptr + compressed_pos[:, None] * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx[None, :], mask=cs_mask, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx[None, :], mask=cs_mask, other=0.0)

    # Only the rope pairs are rotated; the other pairs keep their norm values.
    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = tl.where(is_rope_pair[None, :], even * cos_v - odd * sin_v, even)
    new_odd = tl.where(is_rope_pair[None, :], odd * cos_v + even * sin_v, odd)
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ── Per-token FP8 (single 128-wide block) with one float32 scale ──
    # Interleave the pairs back into the row order for the store.
    result = tl.reshape(tl.join(new_even, new_odd), (RPP, HEAD_SIZE))
    result_bf16 = result
    # Row maximum of |result| on the bit patterns: for finite values the
    # integer order is the magnitude order, inf is 0x7F800000 and NaN lies
    # above it. The floating-point reduction drops NaN in the SIMT lowering.
    result_bits = result_bf16.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    absmax_bits = tl.max(result_bits, axis=1)
    absmax = absmax_bits.to(tl.float32, bitcast=True)
    scale = _pow2_scale(absmax, _SCALE_FLOOR, FP8_MAX)
    x_clamped = tl.clamp(
        result_bf16 * _pow2_recip(scale)[:, None], -FP8_MAX, FP8_MAX, propagate_nan=tl.PropagateNan.ALL
    )
    x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # A NaN row gets the FP8 NaN code (127) and a NaN scale, assembled from its
    # bit pattern. An infinite absmax gives the scale +inf, and the products
    # above quantize as in the reference: 0 for a finite value, NaN for an
    # infinite one.
    row_nan = absmax_bits > 0x7F800000
    x_uint8 = tl.where(row_nan[:, None], 127, x_uint8.to(tl.int32)).to(tl.uint8)
    scale = tl.where(row_nan, 0x7FC00000, scale.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)

    # ── Paged cache pointers (segregated: values first, then scales) ──
    # int32 page offsets: the entry dispatches here only for caches below 2^31 bytes.
    cols = tl.arange(0, HEAD_SIZE)
    page_base = (kv_slot // KV_BLOCK_SIZE) * KV_BLOCK_STRIDE
    row_in_page = kv_slot % KV_BLOCK_SIZE
    tl.store(
        k_cache_ptr + page_base[:, None] + (row_in_page * HEAD_SIZE)[:, None] + cols[None, :],
        x_uint8,
        mask=mask,
    )
    scale_base = (page_base + KV_BLOCK_SIZE * HEAD_SIZE + row_in_page * 4) // 4
    one = tl.arange(0, 1)
    if MX_SCALE:
        tl.store(
            k_cache_ptr.to(tl.pointer_type(tl.int32)) + scale_base[:, None] + one[None, :],
            _pack_ue8m0_x4(scale)[:, None],
            mask=mask,
        )
    else:
        tl.store(
            k_cache_ptr.to(tl.pointer_type(tl.float32)) + scale_base[:, None] + one[None, :],
            scale[:, None],
            mask=mask,
        )


def _indexer_k_norm_rope_store_impl(
    k_pre,
    positions,
    cos_sin_cache,
    rms_norm_weight,
    rms_norm_eps,
    k_cache,
    kv_slot_mapping,
    compress_ratio,
    mx_scale,
):
    num_tokens = kv_slot_mapping.numel()
    head_dim = k_pre.shape[1]
    positions = positions.contiguous()
    kv_slot_mapping = kv_slot_mapping.contiguous()
    if num_tokens >= _TILE_MIN_TOKENS and k_cache.numel() < _INT32_CACHE_BYTES:
        # The pure SIMT lowering runs the whole program on the threads (Ascend
        # 950 and 960).
        tile = _TILE_LARGE if num_tokens >= _TILE_LARGE_MIN_TOKENS else _TILE_SMALL
        _indexer_k_norm_rope_store_rows[((num_tokens + tile["RPP"] - 1) // tile["RPP"],)](
            k_pre,
            k_pre.stride(0),
            positions,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            k_cache,
            kv_slot_mapping,
            num_tokens,
            HEAD_SIZE=head_dim,
            ROPE_HEAD_DIM=64,
            COMPRESS_RATIO=compress_ratio,
            KV_BLOCK_SIZE=k_cache.shape[1],
            KV_BLOCK_STRIDE=k_cache.stride(0),
            FP8_MAX=_FP8_MAX,
            MX_SCALE=mx_scale,
            **tile,
            compile_mode="simt_only",
        )
        return
    # Row-major values (no 16x16 tiled layout); page offsets are int64 scalars.
    _indexer_k_norm_rope_quant_store_kernel[(num_tokens,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=head_dim,
        SCALE_DIM=4,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        FP8_MAX=_FP8_MAX,
        MX_SCALE=mx_scale,
        num_warps=1,
    )


@torch.library.custom_op(
    "mojo_npu_triton_a5::indexer_k_norm_rope_store_fwd",
    mutates_args=("k_cache",),
)
def indexer_k_norm_rope_store_fwd(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    mx_scale: bool,
) -> None:
    _indexer_k_norm_rope_store_impl(
        k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache,
        kv_slot_mapping, compress_ratio, mx_scale,
    )


@indexer_k_norm_rope_store_fwd.register_fake
def _indexer_k_norm_rope_store_fake(
    k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache,
    kv_slot_mapping, compress_ratio, mx_scale,
) -> None:
    return None
