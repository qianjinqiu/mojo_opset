"""combine_topk_swa_indices performance cases (production shapes, cold L2 via the harness)."""

import pytest
import torch

from mojo_opset import functions as F

WINDOW = 128
ALIGN = 128

# (num_reqs, query_len, seq_len, compress_ratio, topk): sparse-prefill chunks
# of up to four requests at index_topk 2048, a continuation chunk (8192 new
# rows after 8192 context rows), the sliding-window-only layer of the same
# chunk, and decode batches with one row per request.
CASES = [
    (1, 8192, 8192, 2, 2048),
    (1, 8192, 8192, 1, 2048),
    (1, 8192, 16384, 2, 2048),
    (4, 2048, 2048, 2, 2048),
    (4, 2048, 2048, 1, 2048),
    (32, 256, 256, 2, 2048),
    (1, 8192, 8192, 0, 0),
    (256, 1, 4096, 2, 2048),
    (256, 1, 4096, 0, 0),
    (64, 1, 4096, 2, 2048),
]


@pytest.mark.api("functions.combine_topk_swa_indices")
@pytest.mark.parametrize(
    "num_reqs,query_len,seq_len,compress_ratio,topk",
    CASES,
    ids=[
        "prefill_8192x1_cr2", "prefill_8192x1_cr1", "chunked_16k_8k_cr2", "chunk_4x2048_cr2",
        "chunk_4x2048_cr1", "chunk_32x256_cr2", "swa_only_8192x1", "mixed_256x1_cr2",
        "mixed_256x1_swa_only", "mixed_64x1_cr2",
    ],
)
def test_combine_topk_swa_indices(benchmark, perf_environment, num_reqs, query_len, seq_len, compress_ratio, topk):
    _, device, _, implementation = perf_environment
    num_tokens = num_reqs * query_len
    row_width = -(-(topk + WINDOW) // ALIGN) * ALIGN
    n_candidates = 0 if topk == 0 else seq_len // compress_ratio

    def factory():
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * query_len
        seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
        gather_lens = torch.minimum(
            seq_lens, torch.full((num_reqs,), query_len + WINDOW - 1, dtype=torch.int32, device=device)
        )
        topk_indices = torch.randint(
            0, max(n_candidates, 1), (num_tokens, topk), dtype=torch.int32, device=device
        )
        topk_indices[:, ::11] = -1
        workspace = (
            torch.empty((num_tokens, row_width), dtype=torch.int32, device=device),
            torch.empty((num_tokens,), dtype=torch.int32, device=device),
        )
        return lambda: F.combine_topk_swa_indices(
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            WINDOW,
            compress_ratio,
            topk,
            n_candidates + query_len + WINDOW - 1,
            n_candidates,
            workspace,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="combine_topk_swa_indices",
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        topk=topk,
        compress_ratio=compress_ratio,
        dtype="int32",
        phase="forward",
    )
