import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.delta_attention import delta_attention_parallel, delta_attention_parallel_nlms, delta_attention_recurrent
from shared.plotting import SEQ_LENS

d_k = 10
d_v = 10

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()
    betas = torch.ones(T)
    outputs1, states1 = delta_attention_parallel(queries, keys, values, betas)
    outputs2, states2 = delta_attention_recurrent(queries, keys, values, betas)

    output_diff = (outputs1 - outputs2).abs().max().item()
    state_diff = (states1 - states2).abs().max().item()

    equal = torch.allclose(outputs1, outputs2, atol=2e-3, rtol=1e-4)

    print(
        f"T={T:4d} | "
        f"equal={equal} | "
        f"max output diff={output_diff:.3e} | "
        f"max state diff={state_diff:.3e}"
    )
