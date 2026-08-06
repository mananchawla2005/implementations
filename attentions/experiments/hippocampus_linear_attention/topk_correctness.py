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

torch.manual_seed(42)
keys = torch.randn(n_items, d_k)
values = torch.randn(n_items, d_v)
scores = torch.randn(n_items)
positions = torch.arange(n_items)

# Inject exact ties across the top-w boundary to exercise the tie policy.
scores[30] = scores[31] = 2.0
scores[70] = scores[71] = 2.0

_, full_idx = scores.topk(w, dim=0, largest=True, sorted=False)
full_set = set(full_idx.tolist())

chunk_size = 7 
pk = torch.empty(0, d_k)
pv = torch.empty(0, d_v)
ps = torch.empty(0)
pp = torch.empty(0, dtype=torch.long)

for start in range(0, n_items, chunk_size):
    end = min(start + chunk_size, n_items)
    pk, pv, ps, pp = update_topk_cache_functional(
        persistent_keys=pk,
        persistent_values=pv,
        persistent_scores=ps,
        persistent_positions=pp,
        new_keys=keys[start:end],
        new_values=values[start:end],
        new_scores=scores[start:end],
        new_positions=positions[start:end],
        capacity=w,
    )

online_set = set(pp.tolist())

print("full top-8 positions :", sorted(full_set))
print("online cache positions:", sorted(online_set))
print("identical selected positions:", full_set == online_set)

full_scores_sorted, _ = scores.topk(w)
cache_scores_sorted = ps.sort(descending=True).values
print("max score-value diff:", (full_scores_sorted - cache_scores_sorted).abs().max().item())

assert full_set == online_set, "online cache diverged from full topk"
assert torch.allclose(full_scores_sorted, cache_scores_sorted, atol=1e-6), "score values diverged"
