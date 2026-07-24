import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_linear_attention import gated_linear_attention

d = 2
T = 5

k1 = torch.tensor([1.0, 0.0])
k2 = torch.tensor([0.0, 1.0])
keys = torch.stack([k1, k2, k1, k2, k1])
queries = keys.clone()

v1 = torch.randn(d)
v2 = torch.randn(d)
values = torch.zeros(T, d)
values[0] = v1
values[1] = v2

alphas = torch.full((T,), 0.2)

_, states = gated_linear_attention(queries, keys, values, alphas)

print(f"v1 (obsolete, written at t=0) = [{v1[0]:+.4f}, {v1[1]:+.4f}]")
print(f"v2 (relevant, written at t=1) = [{v2[0]:+.4f}, {v2[1]:+.4f}]")
print()
print(f"{'t':>2} | {'||col0|| (k1)':>13} | {'||col1|| (k2)':>13} | ratio")
for t in range(T):
    col0_norm = states[t, :, 0].norm().item()
    col1_norm = states[t, :, 1].norm().item()
    ratio = col1_norm / col0_norm if col0_norm > 0 else 0
    print(f"{t:2d} | {col0_norm:>12.6f}  | {col1_norm:>12.6f}  | {ratio:.4f}")
