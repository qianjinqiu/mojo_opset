"""Indexer K store for DeepSeek V4.1 kv-source layers: k_norm, RoPE, FP8 quant."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

HEAD_SIZE = 128
ROPE_DIM = 64
RECORD_BYTES = HEAD_SIZE + 4
_INDEX_DTYPES = (torch.int32, torch.int64)
_SCALE_FORMATS = ("fp32", "mxfp8")


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool = False,
    scale_fmt: str = "fp32",
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Write the k-normalized, rotated and FP8-quantized indexer keys into ``k_cache``.

    Only tokens with ``kv_slot_mapping[t] >= 0`` and
    ``(positions[t] + 1) % compress_ratio == 0`` write; every other token leaves
    the cache untouched. For each written token the row ``k_pre[t]`` goes through
    RMSNorm (FP32, bf16 round trip), a GPT-J interleaved rotation of the last 64
    dims at the group start position (``position // compress_ratio *
    compress_ratio``, bf16 round trip), and a quantization of the whole token
    with one power-of-two scale ``2 ** ceil(log2(max(absmax, 1e-4) / 448))`` to
    ``float8_e4m3fn``. The cache is written in place and returned.

    ``k_pre`` is bf16 ``[rows, 128]`` with unit column stride; the first
    ``kv_slot_mapping.numel()`` rows are read. ``positions`` and
    ``kv_slot_mapping`` are int32/int64 1D; ``positions`` may be longer than
    ``kv_slot_mapping``. ``cos_sin_cache`` is FP32 or bf16 ``[max_pos, 64]``
    holding 32 cos then 32 sin values, with unit column stride;
    ``rms_norm_weight`` is bf16 or FP32 ``[128]``. ``compress_ratio`` is 1 or 2.

    ``k_cache`` is uint8 ``[pages, block_size, 132]`` with unit record stride
    and a 132-byte row stride; its page stride is a multiple of 4, at least
    ``block_size * 132``, and its storage offset is a multiple of 4. A page
    holds ``block_size * 128`` value bytes followed by ``block_size * 4`` scale
    bytes. ``scale_fmt`` selects the scale format: ``"fp32"`` is one FP32 scale,
    ``"mxfp8"`` is four equal UE8M0 bytes. Slots, valid positions of written
    tokens, ``rms_norm_eps`` and a 4-byte aligned start address of ``k_cache``
    are preconditions of the caller.
    """
    _require_inference(
        "indexer_k_norm_rope_store", k_pre, positions, cos_sin_cache, rms_norm_weight, k_cache, kv_slot_mapping
    )
    if use_fp4_cache:
        raise NotImplementedError(
            "indexer_k_norm_rope_store: the MXFP4 indexer K cache has no Ascend equivalent "
            "(upstream packs values with the PTX cvt.rn.satfinite.e2m1x2.f32 instruction)"
        )
    if scale_fmt not in _SCALE_FORMATS:
        raise ValueError(f"scale_fmt must be one of {_SCALE_FORMATS}, got {scale_fmt!r}")
    if k_pre.ndim != 2 or k_pre.shape[1] != HEAD_SIZE or k_pre.dtype != torch.bfloat16:
        raise ValueError("k_pre must be a bf16 [rows, 128] tensor")
    if k_pre.stride(1) != 1:
        raise ValueError("k_pre must have unit column stride")
    if positions.ndim != 1 or positions.dtype not in _INDEX_DTYPES:
        raise ValueError("positions must be a 1D int32/int64 tensor")
    if kv_slot_mapping.ndim != 1 or kv_slot_mapping.dtype not in _INDEX_DTYPES:
        raise ValueError("kv_slot_mapping must be a 1D int32/int64 tensor")
    num_tokens = kv_slot_mapping.numel()
    if num_tokens > k_pre.shape[0] or num_tokens > positions.numel():
        raise ValueError("kv_slot_mapping cannot be longer than k_pre/positions")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != ROPE_DIM:
        raise ValueError("cos_sin_cache must be an [max_pos, 64] tensor")
    if cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("cos_sin_cache must be an fp32/bf16 tensor")
    if cos_sin_cache.stride(1) != 1:
        raise ValueError("cos_sin_cache must have unit column stride")
    if rms_norm_weight.ndim != 1 or rms_norm_weight.shape[0] != HEAD_SIZE:
        raise ValueError("rms_norm_weight must be a [128] tensor")
    if rms_norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("rms_norm_weight must be a bf16/fp32 tensor")
    if rms_norm_weight.stride(0) != 1:
        raise ValueError("rms_norm_weight must be contiguous")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    for name, tensor in (
        ("positions", positions),
        ("cos_sin_cache", cos_sin_cache),
        ("rms_norm_weight", rms_norm_weight),
        ("k_cache", k_cache),
        ("kv_slot_mapping", kv_slot_mapping),
    ):
        if tensor.device != k_pre.device:
            raise ValueError(f"{name} must be on the k_pre device")
    if k_cache.ndim != 3 or k_cache.shape[2] != RECORD_BYTES or k_cache.dtype != torch.uint8:
        raise ValueError("k_cache must be a uint8 [pages, block_size, 132] tensor")
    if k_cache.stride(2) != 1 or k_cache.stride(1) != RECORD_BYTES:
        raise ValueError("k_cache must have unit record stride and a 132-byte row stride")
    if k_cache.stride(0) % 4 or k_cache.stride(0) < k_cache.shape[1] * RECORD_BYTES:
        raise ValueError("k_cache page stride must be a multiple of 4 and at least block_size * 132")

    if num_tokens == 0:
        return k_cache
    forward, _ = load_impl("indexer_k_norm_rope_store", implementation)
    forward(
        k_pre,
        positions,
        cos_sin_cache,
        rms_norm_weight,
        float(rms_norm_eps),
        k_cache,
        kv_slot_mapping,
        int(compress_ratio),
        scale_fmt == "mxfp8",
    )
    return k_cache
