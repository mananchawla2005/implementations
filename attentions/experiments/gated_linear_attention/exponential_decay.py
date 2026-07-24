import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_linear_attention import gated_linear_attention
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.zeros(T, d_v)
    values[0, 0] = 1
    queries = keys.clone()
    alphas = torch.ones(T)*0.9
    gated_outputs, gated_states = gated_linear_attention(queries, keys, values, alphas)
    P = alphas.cumprod(dim=0)
    frobenius_states = gated_states.norm(dim=(1,2))
    equal = torch.allclose(frobenius_states[1:], P[:-1]*frobenius_states[0], atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} "
    )
