import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.delta_attention import delta_attention_parallel

d = 2

ka = torch.tensor([1.0, 0.0])
kb = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
keys = torch.stack([ka, kb, ka])
queries = keys.clone()

va = torch.randn(d)
vb = torch.randn(d)
va2 = torch.randn(d)

values = torch.stack([va, vb, va2])
betas = torch.ones(3)

_, states = delta_attention_parallel(queries, keys, values, betas)

read_vb_before = states[1] @ kb
read_vb_after = states[2] @ kb
change = (read_vb_after - read_vb_before).abs().max().item()
ka_dot_kb = (ka @ kb).item()

print(f"vb                                = [{vb[0]:+.4f}, {vb[1]:+.4f}]")
print(f"Read kb before overwrite ka       = [{read_vb_before[0]:+.4f}, {read_vb_before[1]:+.4f}]")
print(f"Read kb after  overwrite ka       = [{read_vb_after[0]:+.4f}, {read_vb_after[1]:+.4f}]")
print(f"\nChange in kb's value after ka overwrite: {change:.6f}")
