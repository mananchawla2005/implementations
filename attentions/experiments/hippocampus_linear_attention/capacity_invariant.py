import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.hippocampus_linear_attention import update_topk_cache_functional
d_k = 16
d_v = 8
n_items = 100
capacity = 8

torch.manual_seed(0)
keys = torch.randn(n_items, d_k)
values = torch.randn(n_items, d_v)
scores = torch.randn(n_items)
positions = torch.arange(n_items)

# many high scores early so eviction pressure is real
scores[:40] = 3.0
scores[40:80] = -1.0

pk = torch.empty(0, d_k)
pv = torch.empty(0, d_v)
ps = torch.empty(0)
pp = torch.empty(0, dtype=torch.long)

violations = []

for t in range(n_items):
    pk, pv, ps, pp = update_topk_cache_functional(
        persistent_keys=pk,
        persistent_values=pv,
        persistent_scores=ps,
        persistent_positions=pp,
        new_keys=keys[t : t + 1],
        new_values=values[t : t + 1],
        new_scores=scores[t : t + 1],
        new_positions=positions[t : t + 1],
        capacity=capacity,
    )

    if pk.shape[0] > capacity:
        violations.append((t, "size", pk.shape[0]))

    dims = [pk.shape[0], pv.shape[0], ps.shape[0], pp.shape[0]]
    if len(set(dims)) != 1:
        violations.append((t, "dims", dims))

    if (pp > t).any():
        violations.append((t, "future", pp[pp > t].tolist()))

    if len(pp) != len(torch.unique(pp)):
        violations.append((t, "duplicate", pp.tolist()))

print(f"processed {n_items} tokens, capacity = {capacity}")
print(f"violations: {len(violations)}")
for v in violations[:10]:
    print("  ", v)
print(f"final cache size: {pk.shape[0]}")

assert not violations, "capacity invariant violated"
