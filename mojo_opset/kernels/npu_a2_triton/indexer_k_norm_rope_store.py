"""Indexer K store for DeepSeek V4.1 kv-source layers on A2 (910B): unsupported.

The op's only output is an FP8-quantized cache write, and 910B (DAV_2201) has no
FP8 cast, so the entry raises before any launch.
"""

import torch


def _indexer_k_norm_rope_store_impl(k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache,
                                    kv_slot_mapping, compress_ratio, mx_scale):
    raise NotImplementedError(
        "indexer_k_norm_rope_store: 910B (A2/DAV_2201) has no FP8 cast, so the quantised "
        "indexer K record cannot be written on this arch"
    )


@torch.library.custom_op(
    "mojo_npu_triton_a2::indexer_k_norm_rope_store_fwd",
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
        k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache, kv_slot_mapping,
        compress_ratio, mx_scale,
    )


@indexer_k_norm_rope_store_fwd.register_fake
def _indexer_k_norm_rope_store_fake(
    k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache, kv_slot_mapping,
    compress_ratio, mx_scale,
) -> None:
    return None
