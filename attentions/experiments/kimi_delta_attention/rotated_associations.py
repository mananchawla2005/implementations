import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent
from shared.plotting import plot_times

d_k = 2
d_v = 2
T = 30

ka = torch.tensor([1.0, 0.0])
kb = torch.tensor([1.0, 1.0]) / math.sqrt(2)
va = torch.tensor([1.0, 0.0])
vb = torch.tensor([0.0, 1.0])

keys = torch.zeros(T, d_k)
values = torch.zeros(T, d_v)
queries = keys.clone()
keys[0] = ka
keys[1] = kb
values[0] = va
values[1] = vb

alphas = torch.full((T, d_k), 1.0)
alphas[:, 0] = 0.1
alphas[:, 1] = 1.0

betas = torch.ones(T)

_, states = kimi_delta_attention_recurrent(queries, keys, values, alphas, betas)

trace_a = [torch.norm(states[t].T @ ka).item() for t in range(T)]
trace_b = [torch.norm(states[t].T @ kb).item() for t in range(T)]

print(f"ka = [1, 0]")
print(f"kb = [1, 1]/sqrt(2)  ->  ka.kb = {ka @ kb:.4f}")
print(f"alpha = [0.1 (dim0), 1.0 (dim1)]")
print(f"va = [1, 0]")
print(f"vb = [0, 1]")
print()
print(f"{'t':>3} | {'||ka||':>10} | {'||kb||':>10}")
print("-" * 28)
for t in [0, 1, 2, 3, 4, 5, 10, 20, 29]:
    print(f"{t:3d} | {trace_a[t]:>10.6f} | {trace_b[t]:>10.6f}")

plot_times(
    list(range(T)),
    {"ka (damaged by dim0 decay)": trace_a, "kb (collateral damage)": trace_b},
    xlabel="Step",
    ylabel="Retrieval strength",
    title="Rotated Associations: Basis-Dependent Decay",
)
