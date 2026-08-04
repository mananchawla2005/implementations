import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

d_k = 2
d_v = 3
T = 2
W = 2
R = 1
memory_size = 10

torch.manual_seed(0)
queries = torch.randn(T, d_k)
keys = torch.randn(T, d_k)
values = torch.randn(T, d_v)
query_indices = torch.tensor([[0], [0]])
key_indices = torch.tensor([[0, 1], [2, 7]])  # step 1 writes only slots {2, 7}
alphas = torch.ones(T)
betas = torch.ones(T)

outputs, memories = sparse_delta_memory_recurrent(
    queries, query_indices, keys, key_indices, values, alphas, betas, memory_size
)

before = memories[0].clone()  # state after step 0
after = memories[1].clone()   # state after step 1 (writes slots 2, 7)

selected = {2, 7}
unchanged = all(
    torch.equal(before[i], after[i])
    for i in range(memory_size)
    if i not in selected
)

print("Untouched slots bitwise unchanged:", unchanged)

print("\nDirty slots (should have changed):")
for i in sorted(selected):
    changed = not torch.equal(before[i], after[i])
    print(f"  slot {i}: changed={changed}")

assert unchanged, "untouched slots were modified"
for i in sorted(selected):
    assert not torch.equal(before[i], after[i]), f"selected slot {i} did not change"