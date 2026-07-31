import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent
from shared.plotting import plot_times

d_k = 2
d_v = 2
T = 100

ka = torch.tensor([1.0, 0.0])
kb = torch.tensor([0.0, 1.0])
keys = torch.stack([ka, kb] + [ka] * (T - 2))
queries = keys.clone()

va = torch.randn(d_v)
vb = torch.randn(d_v)
values = torch.zeros(T, d_v)
values[0] = va
values[1] = vb

alphas = torch.full((T, d_k), 1.0)
alphas[:, 0] = 0.1
alphas[:, 1] = 0.99
betas = torch.zeros(T)
betas[0] = 1.0
betas[1] = 1.0

_, states = kimi_delta_attention_recurrent(queries, keys, values, alphas, betas)

strength_a = [torch.norm(states[t].T @ ka).item() for t in range(T)]
strength_b = [torch.norm(states[t].T @ kb).item() for t in range(T)]

plot_times(
    list(range(T)),
    {"ka (alpha=0.1)": strength_a, "kb (alpha=0.99)": strength_b},
    xlabel="Step",
    ylabel="Retrieval strength",
    title="Anisotropic Forgetting: Two Memory Timescales",
)
