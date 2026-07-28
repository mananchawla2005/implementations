import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.gated_delta_attention import gated_delta_attention_parallel

d = 2

ka = torch.tensor([1.0, 0.0])
kb = torch.tensor([0.0, 1.0])
keys = torch.stack([ka, kb, ka])
queries = keys.clone()

va = torch.randn(d)
vb = torch.randn(d)
va2 = torch.randn(d)
values = torch.stack([va, vb, va2])

betas = torch.ones(3)
alphas = torch.ones(3)

_, states = gated_delta_attention_parallel(queries, keys, values, betas, alphas)

read_vb_before = states[1] @ kb
read_vb_after = states[2] @ kb

print(f"va  = [{va[0]:+.4f}, {va[1]:+.4f}]")
print(f"vb  = [{vb[0]:+.4f}, {vb[1]:+.4f}]")
print(f"va2 = [{va2[0]:+.4f}, {va2[1]:+.4f}]")
print()
print(f"After writing (va, ka) and (vb, kb):")
print(f"Read kb: [{read_vb_before[0]:+.4f}, {read_vb_before[1]:+.4f}]")
print(f"Match vb: {torch.allclose(read_vb_before, vb, atol=1e-4)}")
print()
print(f"After overwriting ka with va2:")
print(f"Read kb: [{read_vb_after[0]:+.4f}, {read_vb_after[1]:+.4f}]")
print(f"Still matches vb: {torch.allclose(read_vb_after, vb, atol=1e-4)}")
print(f"Diff from original vb: {(read_vb_after - vb).abs().max().item():.3e}")
