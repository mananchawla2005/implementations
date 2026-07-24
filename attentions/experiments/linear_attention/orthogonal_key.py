import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.linear_attention import linear_attention_recurrent
import torch

d = 2

v1 = torch.randn(d)
v2 = torch.randn(d)

k1 = torch.tensor([1.0, 0.0])
k2 = torch.tensor([0.0, 1.0])

values = torch.stack([v1, v2])
keys = torch.stack([k1, k2])
queries = torch.randn(2, d)

_, states = linear_attention_recurrent(queries, keys, values)

read_v1 = states[-1] @ k1
read_v2 = states[-1] @ k2

print(f"v1      = [{v1[0]:.4f}, {v1[1]:.4f}]")
print(f"read k1 = [{read_v1[0]:.4f}, {read_v1[1]:.4f}]")
print(f"Match v1: {torch.allclose(read_v1, v1)} | diff={ (read_v1 - v1).abs().max().item():.3e}")

print(f"\nv2      = [{v2[0]:.4f}, {v2[1]:.4f}]")
print(f"read k2 = [{read_v2[0]:.4f}, {read_v2[1]:.4f}]")
print(f"Match v2: {torch.allclose(read_v2, v2)} | diff={ (read_v2 - v2).abs().max().item():.3e}")
