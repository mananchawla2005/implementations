import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 64
d_v = 8
n_facts = 5
n_noise = 5
n_pred = 50
eta = 2.0
torch.manual_seed(0)

g = torch.Generator().manual_seed(0)


def nk():
    return F.normalize(torch.randn(d_k, generator=g), dim=-1)


facts = [(nk(), torch.randn(d_v, generator=g) * 5.0, 1.0) for _ in range(n_facts)]
noise = [(nk(), torch.randn(d_v, generator=g) * 5.0, 1.0) for _ in range(n_noise)]
pred = [(nk(), torch.randn(d_v, generator=g) * 0.1, 0.02) for _ in range(n_pred)]

seq = (
    pred[:15] + facts[:2] + pred[15:25] + noise[:2]
    + pred[25:35] + facts[2:] + pred[35:40] + noise[2:] + pred[40:]
)

fact_keys = [f[0] for f in facts]
noise_keys = [n[0] for n in noise]

# oracle future utility: facts are queried later, noise is not
utility = {}
for f in facts:
    utility[id(f[0])] = 1.0
for n in noise:
    utility[id(n[0])] = 0.0
for p in pred:
    utility[id(p[0])] = 0.05


def simulate(score_name, w):
    S = torch.zeros(d_k, d_v)
    ck, cv, cs = [], [], []
    for (k, v, b) in seq:
        e = v - S.T @ k
        S = 0.99 * S + b * torch.outer(k, e)
        base = b * e.norm().item()
        if score_name == "m1":
            sc = base
        elif score_name == "m2":
            sc = base * utility[id(k)]
        else:  # m3
            red = (k @ torch.stack(ck).T).max().item() if ck else 0.0
            sc = base - eta * red
        ck.append(k); cv.append(v); cs.append(sc)
        if len(cs) > w:
            idx = min(range(len(cs)), key=lambda j: cs[j])
            del ck[idx], cv[idx], cs[idx]

    fk = sum(1 for k in ck if any(torch.equal(k, f0) for f0 in fact_keys))
    nk_kept = sum(1 for k in ck if any(torch.equal(k, n0) for n0 in noise_keys))
    return fk, nk_kept


print(f"{'w':>3} | {'score':>6} | {'facts kept':>11} | {'noise kept':>11}")
print("-" * 40)
for w in [6, 10]:
    for name in ["m1", "m2", "m3"]:
        fk, nk = simulate(name, w)
        print(f"{w:>3} | {name:>6} | {fk:>5}/{n_facts} | {nk:>5}/{n_noise}")

for w in [6]:
    fk1, nk1 = simulate("m1", w)
    fk2, nk2 = simulate("m2", w)
    assert fk2 > fk1, "utility-aware score should retain more facts"
    assert nk2 < nk1, "utility-aware score should keep less noise"
