import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

d_k = 2
d_v = 4
memory_size = 8
W = 2
R = 2
slots = [0, 1, 2, 3]

T = 5
torch.manual_seed(0)
stored = {i: torch.randn(d_v) for i in slots}

one_hot = torch.tensor([1.0, 0.0])

queries = torch.randn(T, d_k)
query_indices = torch.tensor([[0, 1]] * 4 + [[1, 3]])
keys = torch.stack([one_hot] * 4 + [torch.zeros(d_k)])
values = torch.stack([stored[0], stored[1], stored[2], stored[3], torch.zeros(d_v)])
key_indices = torch.tensor([
    [0, 4],  
    [1, 5], 
    [2, 6],  
    [3, 7], 
    [1, 3], 
])
alphas = torch.ones(T)
betas = torch.ones(T)

betas[4] = 0.0

outputs, memories = sparse_delta_memory_recurrent(
    queries, query_indices, keys, key_indices, values, alphas, betas, memory_size
)

q = queries[4]  
expected = q[0] * stored[1] + q[1] * stored[3]
actual = outputs[4]

print("Read slots {1, 3}, weights:", q.tolist())
print("M[1] =", stored[1])
print("M[3] =", stored[3])
print("Expected weighted sum:", expected)
print("Actual output        :", actual)

matches = torch.allclose(actual, expected, atol=1e-6)
print("Sparse read matches weighted sum:", matches)

assert matches, "sparse read did not equal the weighted sum"
