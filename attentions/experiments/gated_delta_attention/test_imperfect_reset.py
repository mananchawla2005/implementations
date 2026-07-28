import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_delta_attention import gated_delta_attention_parallel
from shared.plotting import plot_times

d = 8
nA = 4
nB = 4

keys_a = torch.eye(d)[:nA]
vals_a = torch.randn(nA, d)

keys_b = torch.eye(d)[nA:nA+nB]
vals_b = torch.randn(nB, d)

keys = torch.cat([keys_a, keys_b])
values = torch.cat([vals_a, vals_b])
queries = keys.clone()

alpha_values = [0.0, 0.1, 0.5, 0.9, 1.0]
R_old = []

for alpha_switch in alpha_values:
    betas = torch.ones(nA + nB)
    alphas = torch.ones(nA + nB)
    alphas[nA] = alpha_switch

    _, states = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

    state_after = states[nA]
    retrievals = state_after @ keys_a.T
    R = retrievals.norm(dim=1).mean().item()
    R_old.append(R)

print(f"{'alpha':>6} | {'R_old':>10}")
print("-" * 20)
for a, r in zip(alpha_values, R_old):
    print(f"{a:>6.1f} | {r:>10.6f}")

plot_times(
    alpha_values,
    {"R_old": R_old},
    xlabel="alpha at document boundary",
    ylabel="Mean norm of old retrievals",
    title="Imperfect Reset: Residual Old-Context Retrieval vs alpha",
)
