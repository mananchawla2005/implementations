import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.linear_attention import linear_attention_recurrent
import torch
import math

d = 2

v1 = torch.randn(d)
v2 = torch.randn(d)

k1 = torch.tensor([1.0, 0.0])
k2 = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])

values = torch.stack([v1, v2])
keys = torch.stack([k1, k2])
queries = torch.randn(2, d)

_, states = linear_attention_recurrent(queries, keys, values)

read_v1 = states[-1] @ k1
read_v2 = states[-1] @ k2

print(f"k1 . k2 = {k1 @ k2:.4f}")
print()

print(f"v1 = [{v1[0]}, {v1[1]}]")
print(f"v2 = [{v2[0]}, {v2[1]}]")
print()

contam_k1 = read_v1 - v1
contam_k2 = read_v2 - v2
expected_contam_k1 = v2 * (k2 @ k1)
expected_contam_k2 = v1 * (k1 @ k2)

print(f"Read k1: [{read_v1[0]}, {read_v1[1]}]")
print(f"Desired: v1                 = [{v1[0]}, {v1[1]}]")
print(f"Contamination from v2:      [{contam_k1[0]}, {contam_k1[1]}]")
print(f"Expected v2 * (k2.k1):      [{expected_contam_k1[0]}, {expected_contam_k1[1]}]")
print(f"Contamination match: {torch.allclose(contam_k1, expected_contam_k1)}")
print()

print(f"Read k2: [{read_v2[0]}, {read_v2[1]}]")
print(f"Desired: v2                 = [{v2[0]}, {v2[1]}]")
print(f"Contamination from v1:      [{contam_k2[0]}, {contam_k2[1]}]")
print(f"Expected v1 * (k1.k2):      [{expected_contam_k2[0]}, {expected_contam_k2[1]}]")
print(f"Contamination match: {torch.allclose(contam_k2, expected_contam_k2)}")
