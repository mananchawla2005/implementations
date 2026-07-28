import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_delta_attention import gated_delta_attention_parallel

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

betas = torch.ones(nA + nB)
alphas = torch.ones(nA + nB)
alphas[nA] = 0.0

_, states = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

state_after = states[nA]
retrievals = state_after @ keys_a.T
expected = torch.zeros_like(retrievals)

print("Document A (first 4 one-hot keys, doc B uses keys 4-7):")
for i in range(nA):
    print(f"  v{i} = [{vals_a[i,0]:+.3f}, {vals_a[i,1]:+.3f}, {vals_a[i,2]:+.3f}, {vals_a[i,3]:+.3f}, ...]")
print()

print(f"After alpha=0 boundary, reading doc A keys:")
for i in range(nA):
    r = retrievals[i]
    print(f"  k{i}: [{r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}, {r[3]:+.4f}, ...]")
print()

all_zero = torch.allclose(retrievals, expected, atol=1e-4)
residual = retrievals.abs().max().item()
print(f"All doc A retrievals near zero after reset: {all_zero}")
print(f"Max residual old retrieval: {residual:.3e}")
