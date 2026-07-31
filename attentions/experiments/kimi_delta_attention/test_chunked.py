import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent
from algorithms.kimi_delta_attention import kimi_delta_attention_chunked
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    alphas = torch.ones(T, d_k)
    betas = torch.ones(T)
    chunked_out = kimi_delta_attention_chunked(queries, keys, values, betas, alphas, chunk_size=10)
    recurrent_out, _ = kimi_delta_attention_recurrent(queries, keys, values, betas, alphas)

    output_diff = (chunked_out - recurrent_out).abs().max().item()

    equal = torch.allclose(chunked_out, recurrent_out, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e}"
    )
