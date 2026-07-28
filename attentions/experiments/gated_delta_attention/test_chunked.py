import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_delta_attention import gated_delta_attention_parallel
from algorithms.gated_delta_attention import gated_delta_attention_chunked
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    alphas = torch.ones(T)
    betas = torch.ones(T)
    chunked_out = gated_delta_attention_chunked(queries, keys, values, betas, alphas, chunk_size=10)
    gdn_out, _ = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

    output_diff = (chunked_out - gdn_out).abs().max().item()

    equal = torch.allclose(chunked_out, gdn_out, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e}"
    )
