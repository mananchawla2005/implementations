import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.delta_attention import delta_attention_parallel

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

_, states = delta_attention_parallel(queries, keys, values, betas)

read_vb_after_a = states[1] @ kb
read_vb_after_overwrite = states[2] @ kb

print(f"va  = [{va[0]}, {va[1]}]")
print(f"vb  = [{vb[0]}, {vb[1]}]")
print(f"va2 = [{va2[0]}, {va2[1]}]")
print()
print(f"Read kb: [{read_vb_after_a[0]}, {read_vb_after_a[1]}]")
print(f"Match vb: {torch.allclose(read_vb_after_a, vb)}")
print()
print(f"After overwriting:")
print(f"Read kb: [{read_vb_after_overwrite[0]}, {read_vb_after_overwrite[1]}]")
print(f"Still matches vb: {torch.allclose(read_vb_after_overwrite, vb)}")
print(f"Diff from original: {(read_vb_after_overwrite - vb).abs().max().item()}")
