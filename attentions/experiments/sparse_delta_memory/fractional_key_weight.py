import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

d_k = 2
d_v = 3
T = 2
W = 2
R = 1
memory_size = 6
slot = 2  
partner = 3  # empty slot, absorbs the other weight

torch.manual_seed(0)
values = torch.randn(T, d_v)

# Key whose first component normalizes to 0.25.
k_raw = torch.tensor([0.25, (1 - 0.25**2) ** 0.5])
kn = F.normalize(k_raw.unsqueeze(0), 2, 1)[0]
print("normalized key:", kn.tolist())
assert torch.allclose(kn[0], torch.tensor(0.25), atol=1e-6)

keys = torch.stack([
    torch.tensor([1.0, 0.0]),
    k_raw,
])
key_indices = torch.tensor([[slot, partner], [slot, partner]])
query_indices = torch.tensor([[slot], [slot]])
queries = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])])
alphas = torch.ones(T)
betas = torch.ones(T)

outputs, memories = sparse_delta_memory_recurrent(
    queries, query_indices, keys, key_indices, values, alphas, betas, memory_size
)

old = memories[0, slot].clone()         
decayed = alphas[0] * old              
expected = decayed + kn[0] * betas[1] * (values[1] - kn[0] * decayed)
actual = memories[1, slot]

print("Old value M_{0}[i]  :", old)
print("Target value v_t    :", values[1])
print("Expected (k=0.25)   :", expected)
print("Actual M_t[i]       :", actual)

matches = torch.allclose(actual, expected, atol=1e-6)
moved_partially = 0 < (actual - old).norm().item() < (values[1] - old).norm().item()

print("\nMatches fractional formula:", matches)
print("Moved only partially toward v_t:", moved_partially)

assert matches, "slot did not follow the fractional-key formula"
assert moved_partially, "slot did not move partially toward v_t"
