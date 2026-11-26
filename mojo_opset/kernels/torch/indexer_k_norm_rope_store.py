"""Eager reference for the indexer K store: k_norm, RoPE, FP8 quant, paged write.

Only the group-tail tokens with a valid slot are written, into the value bytes
and the scale bytes of their slot; every other byte of the cache keeps its
content. The cache is addressed page by page because the page stride may include
padding.
"""

import torch

FP8_MAX = 448.0
SCALE_FLOOR = 1e-4
FP8_DTYPE = torch.float8_e4m3fn
HEAD_SIZE = 128
ROPE_DIM = 64
SCALE_BYTES = 4


def _selection(positions, slot_mapping, compress_ratio):
    valid = (slot_mapping >= 0) & ((positions + 1) % compress_ratio == 0)
    return valid.nonzero().flatten()


def indexer_k_norm_rope_store_fwd(
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
    if num_tokens == 0:
        return None
    index = _selection(positions[:num_tokens], kv_slot_mapping, compress_ratio)
    if index.numel() == 0:
        return None
    block = k_cache.shape[1]
    slot = kv_slot_mapping[index]
    page = (slot // block).long()
    row = (slot % block).long()

    # k_norm: FP32 with a bf16 round trip.
    k = k_pre[index].float()
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    k = (k * rms_norm_weight.float()).to(torch.bfloat16).float()

    # GPT-J interleaved RoPE on the last 64 dims, at the group start position.
    position = positions[index]
    cs_rows = cos_sin_cache[position // compress_ratio * compress_ratio].float()
    cos, sin = cs_rows[:, : ROPE_DIM // 2], cs_rows[:, ROPE_DIM // 2 :]
    even, odd = k[:, ROPE_DIM::2], k[:, ROPE_DIM + 1 ::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(1)
    k = torch.cat((k[:, :ROPE_DIM], rotated), dim=-1).to(torch.bfloat16).float()

    # One power-of-two scale per token; the division is exact for a power of two.
    floor = torch.tensor(SCALE_FLOOR, dtype=torch.float32, device=k.device)
    absmax = torch.maximum(k.abs().amax(dim=-1, keepdim=True), floor)
    scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX))))
    values = (k / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(torch.uint8)
    if mx_scale:
        ue8m0 = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        # Materialized: index_put with a stride-0 source faults on the NPU backend.
        scale_bytes = ue8m0.repeat(1, SCALE_BYTES)
    else:
        scale_bytes = scale.view(torch.uint8).reshape(-1, SCALE_BYTES)

    pages = torch.as_strided(
        k_cache, (k_cache.shape[0], k_cache.stride(0)), (k_cache.stride(0), 1), k_cache.storage_offset()
    )
    value_offset = row[:, None] * HEAD_SIZE + torch.arange(HEAD_SIZE, device=k.device)
    scale_offset = (
        block * HEAD_SIZE + row[:, None] * SCALE_BYTES + torch.arange(SCALE_BYTES, device=k.device)
    )
    # Write through uint8: the NPU backend has no fp8 element-wise index_put.
    pages[page[:, None], value_offset] = values
    pages[page[:, None], scale_offset] = scale_bytes
    return None
