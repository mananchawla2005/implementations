import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.delta_attention import delta_attention_parallel

d = 4

key = torch.ones(d)
keys = key.unsqueeze(0).expand(2, d)
keys = F.normalize(keys, 2, 1)
key = keys[0]
queries = keys.clone()

v = torch.randn(d)
values = torch.stack([v, v])  # same value twice
betas = torch.ones(2)

_, states = delta_attention_parallel(queries, keys, values, betas)

state_unchanged = torch.allclose(states[0], states[1])
delta = (states[1] - states[0]).abs().max().item()
second_excess = (states[1] @ key - v).abs().max().item()

print(f"v = [{', '.join(f'{x:.4f}' for x in v)}]")
print(f"State unchanged after second write: {state_unchanged}")
print(f"Max state change: {delta:.3e}")
print(f"excess = {second_excess:.3e}")
