import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 64
d_v = 8
scales = [1 / (d_k ** 0.5), 0.5, 1.0, 2.0, d_k ** 0.5, 2 * d_k ** 0.5]

torch.manual_seed(0)
g = torch.Generator().manual_seed(0)

k_star = F.normalize(torch.randn(d_k, generator=g), dim=-1)
v_star = torch.randn(d_v, generator=g)
near = F.normalize(0.85 * k_star + 0.15 * torch.randn(d_k, generator=g), dim=-1)
v_near = torch.randn(d_v, generator=g)
distractors = F.normalize(torch.randn(48, d_k, generator=g), dim=-1)

K = torch.cat([k_star.unsqueeze(0), near.unsqueeze(0), distractors])
V = torch.cat([v_star.unsqueeze(0), v_near.unsqueeze(0), torch.randn(48, d_v, generator=g)])
assert K.norm(dim=1).max().item() < 1.001, "keys must be unit-normed"

q_perfect = k_star
# two noisy queries: moderate (target still argmax) and heavy (argmax flips)
g2 = torch.Generator().manual_seed(3)
q_mod = F.normalize(k_star + 0.5 * torch.randn(d_k, generator=g2), dim=-1)
q_heavy = F.normalize(k_star + 2.0 * torch.randn(d_k, generator=g2), dim=-1)

print(f"moderate noise argmax: {'target' if (q_mod @ K.T).argmax() == 0 else 'WRONG'}")
print(f"heavy noise argmax:    {'target' if (q_heavy @ K.T).argmax() == 0 else 'WRONG'}")
print()

def attend(q, c):
    w = torch.softmax(c * c * (q @ K.T) / (d_k ** 0.5), -1)
    return w, w @ V

print(f"{'c':>8} | {'c^2':>7} | {'entropy':>8} | {'max w':>7} | {'perfect':>8} | {'moderate':>9} | {'heavy':>7}")
print("-" * 68)
for c in scales:
    w, retr = attend(q_perfect, c)
    ent = (-(w * torch.log(w + 1e-12)).sum()).item()
    err_perfect = (retr - v_star).norm().item()
    err_mod = (attend(q_mod, c)[1] - v_star).norm().item()
    err_heavy = (attend(q_heavy, c)[1] - v_star).norm().item()
    print(f"{c:>8.3f} | {c*c:>7.2f} | {ent:>8.3f} | {w.max().item():>7.3f} | {err_perfect:>8.3f} | {err_mod:>9.3f} | {err_heavy:>7.3f}")

errs = {c: (attend(q_perfect, c)[1] - v_star).norm().item() for c in scales}
assert errs[2 * d_k ** 0.5] < 0.01, "max sharpening is best on the perfect query"
mod = {c: (attend(q_mod, c)[1] - v_star).norm().item() for c in scales}
assert mod[2 * d_k ** 0.5] < mod[d_k ** 0.5], "sharpening helps moderate noise too"
heavy = {c: (attend(q_heavy, c)[1] - v_star).norm().item() for c in scales}
assert heavy[2 * d_k ** 0.5] > heavy[1 / (d_k ** 0.5)], "brittleness only when argmax flips"
