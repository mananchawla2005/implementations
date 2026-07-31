import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.delta_attention import delta_attention_parallel
from algorithms.gated_delta_attention import gated_delta_attention_parallel
from algorithms.kimi_delta_attention import kimi_delta_attention_chunked
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    scalar_alphas = torch.ones(T)
    betas = torch.ones(T)
    alphas = scalar_alphas[:, None].expand(-1, d_k)
    kimi_delta_out = kimi_delta_attention_chunked(queries, keys, values, betas, alphas, chunk_size=10)
    gdn_out, _ = gated_delta_attention_parallel(queries, keys, values, betas, scalar_alphas)

    output_diff = (kimi_delta_out - gdn_out).abs().max().item()

    equal = torch.allclose(kimi_delta_out, gdn_out, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e} | "
    )
