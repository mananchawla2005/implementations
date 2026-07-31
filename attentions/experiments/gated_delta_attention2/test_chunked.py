import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_delta_attention2 import gated_delta_attention2_recurrent
from algorithms.gated_delta_attention2 import gated_delta_attention2_chunked
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    alphas = torch.ones(T, d_k)
    betas = torch.ones(T)
    erase_gates = betas[:, None].expand(-1, d_k)
    write_gates = betas[:, None].expand(-1, d_v)
    chunked_out = gated_delta_attention2_chunked(queries, keys, values, alphas, erase_gates, write_gates, chunk_size=10)
    recurrent_out, _ = gated_delta_attention2_recurrent(queries, keys, values, alphas, erase_gates, write_gates)

    output_diff = (chunked_out - recurrent_out).abs().max().item()

    equal = torch.allclose(chunked_out, recurrent_out, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e}"
    )
