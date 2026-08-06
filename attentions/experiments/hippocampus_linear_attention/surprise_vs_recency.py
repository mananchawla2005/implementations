import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 64
d_v = 8
capacity = 64
distances = [16, 64, 128, 256, 512]
n_seeds = 5
torch.manual_seed(0)


def rms_norm(x):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8)


def run(D, method, seed):
    g = torch.Generator().manual_seed(seed)
    k_star = F.normalize(torch.randn(d_k, generator=g), dim=-1)
    v_star = torch.randn(d_v, generator=g) * 5.0 

    T = 6 + D
    keys = torch.randn(T, d_k, generator=g)
    keys = F.normalize(keys, dim=-1)
    vals = torch.randn(T, d_v, generator=g) * 0.3       
    betas = torch.full((T,), 0.05)
    keys[5] = k_star
    vals[5] = v_star
    betas[5] = 1.0                                      
    alphas = torch.full((T,), 0.99)

    if method == "full":
        logits = k_star @ keys.T / (d_k ** 0.5)
        w = torch.softmax(logits, -1)
        return (w @ vals - v_star).norm().item()

    S = torch.zeros(d_k, d_v)
    cache_k, cache_v, cache_s = [], [], []
    for t in range(T):
        k, v, a, b = keys[t], vals[t], alphas[t], betas[t]
        e = v - S.T @ k
        S = a * S + b * torch.outer(k, e)
        if method != "state":
            score = {
                "recency": t,
                "resid": e.norm().item(),
                "bv": b * v.norm().item(),
                "be": b * e.norm().item(),
            }[method]
            cache_k.append(k)
            cache_v.append(v)
            cache_s.append(score)
            if len(cache_s) > capacity:
                idx = cache_s.index(min(cache_s))
                del cache_k[idx], cache_v[idx], cache_s[idx]

    q = k_star
    if method == "state":
        return (S.T @ q - v_star).norm().item()

    CK = rms_norm(torch.stack(cache_k))
    CV = torch.stack(cache_v)
    logits = rms_norm(q) @ CK.T / (d_k ** 0.5)  
    w = torch.softmax(logits, -1)
    return (w @ CV - v_star).norm().item()


methods = ["state", "recency", "resid", "bv", "be", "full"]
labels = ["state", "recency", "residual", "beta||v||", "beta||e||", "full-attn"]

print(f"{'distance':>8} |" + "".join(f"{lab:>11}" for lab in labels))
print("-" * 78)
for D in distances:
    row = f"{D:>8} |"
    for m in methods:
        e = sum(run(D, m, s) for s in range(n_seeds)) / n_seeds
        row += f"{e:>11.2f}"
    print(row)

for D in [128, 256, 512]:
    state = sum(run(D, "state", s) for s in range(n_seeds)) / n_seeds
    rec = sum(run(D, "recency", s) for s in range(n_seeds)) / n_seeds
    res = sum(run(D, "resid", s) for s in range(n_seeds)) / n_seeds
    assert state > 2.0, "state should degrade with distance"
    assert rec > 5.0, "recency should evict the far fact"
    assert res < 1.0, "residual cache should retain the fact"
