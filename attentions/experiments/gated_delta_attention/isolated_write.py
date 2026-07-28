import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.gated_delta_attention import gated_delta_attention_parallel

d = 4

key = torch.ones(d)
keys = key.unsqueeze(0).expand(2, d)
keys = F.normalize(keys, 2, 1)
key = keys[0]
queries = keys.clone()

v1 = torch.randn(d)
v2 = torch.randn(d)
values = torch.stack([v1, v2])
betas = torch.ones(2)
alphas = torch.zeros_like(betas)

_, states = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

reads = torch.einsum('ti,tj->tij', values, keys) 

print(f"reads[0] = {reads[0]}")
print(f"states[0] = {states[0]}")
print(f"Match v1: {torch.allclose(reads[0], states[0], atol=1e-4)}")
print()
print(f"reads[1] = {reads[1]}")
print(f"states[1] = {states[1]}")
print(f"Match v1: {torch.allclose(reads[1], states[1], atol=1e-4)}")
