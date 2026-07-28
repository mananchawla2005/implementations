import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.delta_attention import delta_attention_parallel
from algorithms.gated_delta_attention import gated_delta_attention_parallel
from algorithms.delta_attention import delta_attention_parallel
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    alphas = torch.ones(T)
    betas = torch.ones(T)
    delta_out, delta_states = delta_attention_parallel(queries, keys, values, betas)
    gdn_out, parallel_states = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

    output_diff = (delta_out - gdn_out).abs().max().item()
    state_diff = (delta_states - parallel_states).abs().max().item()

    equal = torch.allclose(delta_out, gdn_out, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e} | "
        f"max state diff={state_diff:.3e}"
    )
