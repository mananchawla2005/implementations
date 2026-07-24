import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.linear_attention import linear_attention_recurrent
import torch

d = 8

values = torch.randn(5, d)
queries = torch.randn(5, d)
key = torch.zeros(d)
key[0] = 1.0

keys = key.repeat(5, 1)

_, states = linear_attention_recurrent(queries, keys, values)

output = states[-1] @ key

sum_values = values.sum(dim=0)

print("Values written:")
for i, v in enumerate(values):
    print(f"  v[{i}] = [{v}]")

print(f"\nSum of all values:     [{sum_values}]")
print(f"Read back (S @ key):   [{output}]")
print(f"\nMatch: {torch.allclose(output, sum_values)}")
print(f"Max diff: {(output - sum_values).abs().max().item():.3e}")
