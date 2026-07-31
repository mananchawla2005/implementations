import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent
from shared.plotting import plot_times

d_k = 4
d_v = 4
T = 500

keys = torch.zeros(T, d_k)
values = torch.zeros(T, d_v)
keys[:4] = torch.eye(d_k)
values[:4] = torch.eye(d_k)
queries = keys.clone()

alphas = torch.full((T, d_k), 1.0)
alphas[:, 0] = 0.5
alphas[:, 1] = 0.9
alphas[:, 2] = 0.99
alphas[:, 3] = 0.999

betas = torch.zeros(T)
betas[:4] = 1.0

_, states = kimi_delta_attention_recurrent(queries, keys, values, alphas, betas)

half_lives = [math.log(0.5) / math.log(a) for a in [0.5, 0.9, 0.99, 0.999]]

traces = {}
for i in range(d_k):
    trace = [torch.norm(states[t].T @ keys[i]).item() for t in range(T)]
    traces[f"ch{i} alpha={alphas[0,i].item():.3f}"] = trace

print(f"{'channel':>12} | {'alpha':>6} | {'half-life':>20}")
print("-" * 65)
for i in range(d_k):
    hl = half_lives[i]
    print(f"{'ch' + str(i):>12} | {alphas[0,i].item():>6.3f} | {hl:>17.2f}")

plot_times(
    list(range(T)),
    traces,
    xlabel="Step",
    ylabel="Retrieval strength",
    title="Multi-Timescale Memory (4 channels)",
)
