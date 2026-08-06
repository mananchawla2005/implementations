import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

# HOLA-F: redundant cache entries.
#
# Generate several high-surprise items with nearly identical keys and values.
# Plain top-w surprise may retain all of them, wasting capacity. Measure cache
# diversity with
#     D_K = 2/(w(w-1)) * sum_{i<j} (1 - k_i^T k_j)
# and compare top-w surprise against diversity-aware selection
#     m_i' = m_i - rho * max_{j in A} k_i^T k_j
#
# Research question: should an episodic cache preserve the most surprising
# tokens, or the most surprising NON-redundant tokens?

d_k = 64
d_v = 8
w = 4
rho = 3.0
torch.manual_seed(0)
g = torch.Generator().manual_seed(0)


def nk():
    return F.normalize(torch.randn(d_k, generator=g), dim=-1)


# 3 redundant, 4 diverse, all high surprise
base = nk()
redundant = [F.normalize(base + 0.05 * torch.randn(d_k, generator=g), dim=-1) for _ in range(3)]
diverse = [nk() for _ in range(4)]
items = [(k, torch.randn(d_v, generator=g) * 5.0) for k in redundant]
items += [(k, torch.randn(d_v, generator=g) * 5.0) for k in diverse]
scores = torch.tensor([5.0] * 3 + [4.0] * 4)  # redundant are the top surprises


def diversity(keys):
    n = len(keys)
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 - (keys[i] @ keys[j]).item()
            cnt += 1
    return s / cnt


top_idx = scores.topk(w).indices
plain_keys = torch.stack([items[i][0] for i in top_idx])
plain_red = sum(1 for i in top_idx if i < 3)
plain_div = sum(1 for i in top_idx if i >= 3)

sel_idx = []
pool = list(range(len(items)))
while len(sel_idx) < w and pool:
    best, bestv = None, -1e9
    for i in pool:
        k = items[i][0]
        pen = (rho * (torch.stack([items[j][0] for j in sel_idx]) @ k).max().item()
               if sel_idx else 0.0)
        v = scores[i].item() - pen
        if v > bestv:
            bestv, best = v, i
    sel_idx.append(best)
    pool.remove(best)
div_keys = torch.stack([items[i][0] for i in sel_idx])
div_red = sum(1 for i in sel_idx if i < 3)
div_div = sum(1 for i in sel_idx if i >= 3)

print(f"plain top-w surprise    : D_K = {diversity(plain_keys):.4f}  "
      f"({plain_red} redundant + {plain_div} diverse)")
print(f"diversity-aware         : D_K = {diversity(div_keys):.4f}  "
      f"({div_red} redundant + {div_div} diverse)")

assert diversity(div_keys) > diversity(plain_keys), "diversity-aware should be more diverse"
assert div_red < plain_red, "diversity-aware should keep fewer redundant items"
