"""indexer_k_norm_rope_store performance cases (production shapes, cold L2 via the harness)."""

import pytest
import torch

from mojo_opset import functions as F

HEAD_SIZE = 128
ROPE_DIM = 64
RECORD_BYTES = HEAD_SIZE + 4
BLOCK_SIZE = 64
MAX_POS = 8192
EPS = 1e-20

# (num_tokens, compress_ratio, scale_fmt): a prefill chunk of 8192 tokens at
# both compress ratios and both scale formats, smaller chunks, and the decode
# batch sizes.
CASES = [
    (8192, 2, "fp32"),
    (8192, 1, "fp32"),
    (8192, 2, "mxfp8"),
    (4096, 2, "fp32"),
    (2048, 2, "fp32"),
    (1024, 2, "fp32"),
    (256, 2, "fp32"),
    (64, 2, "fp32"),
    (32, 2, "fp32"),
    (1, 2, "fp32"),
]


@pytest.mark.api("functions.indexer_k_norm_rope_store")
@pytest.mark.parametrize(
    "num_tokens,compress_ratio,scale_fmt",
    CASES,
    ids=[
        "prefill_8192_cr2", "prefill_8192_cr1", "prefill_8192_cr2_mxfp8", "chunk_4096_cr2",
        "chunk_2048_cr2", "chunk_1024_cr2", "t256_cr2", "decode_64_cr2", "decode_32_cr2", "decode_1_cr2",
    ],
)
def test_indexer_k_norm_rope_store(benchmark, perf_environment, num_tokens, compress_ratio, scale_fmt):
    _, device, _, implementation = perf_environment
    pages = 2 * num_tokens // BLOCK_SIZE + 1

    def factory():
        angles = torch.randn(MAX_POS, ROPE_DIM // 2, device=device)
        k_pre = torch.randn(num_tokens, HEAD_SIZE, device=device).to(torch.bfloat16)
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        rms_norm_weight = torch.randn(HEAD_SIZE, device=device).to(torch.bfloat16)
        positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device)
        slots = torch.randperm(pages * BLOCK_SIZE, device=device)[:num_tokens].to(torch.int64).clone()
        k_cache = torch.empty((pages, BLOCK_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device)
        return lambda: F.indexer_k_norm_rope_store(
            k_pre,
            positions,
            cos_sin_cache,
            rms_norm_weight,
            EPS,
            k_cache,
            slots,
            compress_ratio,
            scale_fmt=scale_fmt,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="indexer_k_norm_rope_store",
        num_tokens=num_tokens,
        compress_ratio=compress_ratio,
        scale_fmt=scale_fmt,
        dtype="uint8",
        phase="forward",
    )
