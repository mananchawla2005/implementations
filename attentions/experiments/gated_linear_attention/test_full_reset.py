import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_linear_attention import gated_linear_attention
from shared.plotting import SEQ_LENS

d_k = 4
d_v = 4

key = torch.zeros(d_k)
key[0] = 1.0
query = key.clone()

# decay->write->read convention
print("Changing any value before r has zero effect on outputs[t>=r].\n")
print(f"{'T':>4} | {'r':>4} | {'pre-reset diff':>14} | {'post-reset diff':>15} | {'output[r]==v[r]':>14} | {'post-reset==cumsum(v[r:])':>24}")

for T in SEQ_LENS:
    values = torch.randn(T, d_v)
    keys = key.unsqueeze(0).expand(T, d_k)
    queries = query.unsqueeze(0).expand(T, d_k)

    alphas = torch.ones(T)
    r = T // 2
    alphas[r] = 0.0

    outputs, _ = gated_linear_attention(queries, keys, values, alphas)

    values2 = values.clone()
    values2[0] = torch.randn(d_v)
    outputs2, _ = gated_linear_attention(queries, keys, values2, alphas)

    diff_pre = (outputs[:r] - outputs2[:r]).abs().max().item()
    diff_post = (outputs[r:] - outputs2[r:]).abs().max().item()

    cumulative_from_r = values[r:].cumsum(dim=0)

    match_cumsum = torch.allclose(outputs[r:], cumulative_from_r, atol=1e-4, rtol=1e-4)
    match_v_r = torch.allclose(outputs[r], values[r], atol=1e-5, rtol=1e-5)

    print(
        f"{T:4d} | {r:4d} | {diff_pre:>13.3e} | {diff_post:>14.3e} | {str(match_v_r):>14} | {str(match_cumsum):>24}"
    )
