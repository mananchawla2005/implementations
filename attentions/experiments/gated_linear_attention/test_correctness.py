import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.linear_attention import linear_attention_parallel
from algorithms.gated_linear_attention import gated_linear_attention
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    alphas = torch.ones(T)
    gated_outputs, gated_states = gated_linear_attention(queries, keys, values, alphas)
    parallel_outputs, parallel_states = linear_attention_parallel(queries, keys, values)

    output_diff = (gated_outputs - parallel_outputs).abs().max().item()
    state_diff = (gated_states - parallel_states).abs().max().item()

    equal = torch.allclose(gated_outputs, parallel_outputs, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e} | "
        f"max state diff={state_diff:.3e}"
    )
