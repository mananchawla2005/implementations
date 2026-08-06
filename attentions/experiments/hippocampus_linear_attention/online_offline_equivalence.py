import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.hippocampus_linear_attention import update_topk_cache_functional

d_k = 16
d_v = 8
n_items = 100
w = 8

torch.manual_seed(7)
keys = torch.randn(n_items, d_k)
values = torch.randn(n_items, d_v)
scores = torch.randn(n_items)
positions = torch.arange(n_items)

scores[25] = scores[26] = 2.0
scores[80] = 5.0

_, offline_idx = scores.topk(w, dim=0, largest=True, sorted=False)
offline_set = set(offline_idx.tolist())

pk = torch.empty(0, d_k)
pv = torch.empty(0, d_v)
ps = torch.empty(0)
pp = torch.empty(0, dtype=torch.long)

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
        capacity=w,
    )

online_set = set(pp.tolist())

print("offline top-8:", sorted(offline_set))
print("online  top-8:", sorted(online_set))
print("sets agree:", online_set == offline_set)

print("late high-score token kept:", 80 in online_set)

assert online_set == offline_set, "online/offline sets diverged"
assert 80 in online_set, "online cache missed the high-score token"
