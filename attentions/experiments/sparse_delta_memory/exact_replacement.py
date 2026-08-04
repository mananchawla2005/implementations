import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

d_k = 2
d_v = 4
T = 2
W = 2
R = 1
memory_size = 8
slot = 3

torch.manual_seed(0)
queries = torch.randn(T, d_k)
values = torch.randn(T, d_v)
query_indices = torch.tensor([[slot], [slot]])
alphas = torch.ones(T)
betas = torch.ones(T)

key_indices = torch.tensor([[slot, slot + 1], [slot, slot + 1]])

k_onehot = torch.zeros(d_k)
k_onehot[0] = 1.0
keys = torch.stack([k_onehot, k_onehot])

outputs, memories = sparse_delta_memory_recurrent(
    queries, query_indices, keys, key_indices, values, alphas, betas, memory_size
)

after = memories[1]
matches = torch.allclose(after[slot], values[1], atol=1e-6)

print("Stored value in slot", slot, ":", after[slot])
print("Target value v_t          :", values[1])
print("Exact replacement:", matches)

assert matches, "selected slot did not match v_t exactly"