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
slot = 1

torch.manual_seed(0)

q = torch.randn(T, d_k)
v = torch.randn(T, d_v)
keys = torch.randn(T, d_k)
kidx = torch.tensor([[1, 2], [1, 1]])
qidx = torch.tensor([[1], [1]])
alphas = torch.ones(T)
betas = torch.ones(T)

outputs, memories = sparse_delta_memory_recurrent(
    q, qidx, keys, kidx, v, alphas, betas, memory_size
)

# Reference: last-wins scatter.
M1_prev = memories[0, slot].clone()
kn = F.normalize(keys[1].unsqueeze(0), 2, 1)[0]
pred = (kn[0] + kn[1]) * M1_prev  # prediction sums BOTH duplicate entries
err = v[1] - pred
expected_last_wins = M1_prev + kn[1] * err  # only the second weight is written
actual = memories[1, slot]

print("Slot 1 before collision:", M1_prev)
print("Slot 1 after collision :", actual)
print("Prediction double-counted (k0+k1) * M:", pred)

matches = torch.allclose(actual, expected_last_wins, atol=1e-6)
slot2_untouched = torch.allclose(memories[1, 2], memories[0, 2])

print("\nLast-wins scatter (not combined):", matches)
print("Slot 2 untouched:", slot2_untouched)

combine = M1_prev + (kn[0] + kn[1]) * (v[1] - (kn[0] + kn[1]) * M1_prev)
print("If weights were combined, slot 1 would be:", combine)
print("(differs from actual -> scatter does not combine duplicates)")

assert matches, "collision did not behave as last-wins scatter"
assert slot2_untouched, "unselected slot modified"
