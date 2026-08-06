import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.inference import SparseDeltaMemory, HippocampusLinearAttention

n_seeds = 5
distances = [32, 128, 256]
torch.manual_seed(0)


def build(seed, D, d_k):
    g = torch.Generator().manual_seed(seed)
    k_star = F.normalize(torch.randn(d_k, generator=g), dim=-1)
    v_star = torch.randn(8, generator=g) * 5.0
    T = 5 + D
    keys = F.normalize(torch.randn(T, d_k, generator=g), dim=-1)
    vals = torch.randn(T, 8, generator=g) * 0.3
    keys[0] = k_star
    vals[0] = v_star
    betas = torch.full((T,), 0.05)
    betas[0] = 1.0
    return k_star, v_star, keys, vals, betas


def run_sdm(D, N, seed):
    d_k = 8
    W = R = 8
    k_star, v_star, keys, vals, betas = build(seed, D, d_k)
    P = torch.randn(d_k, N, generator=torch.Generator().manual_seed(seed))
    m = SparseDeltaMemory(N, 8)
    for t in range(len(keys)):
        scores = keys[t] @ P
        _, wi = scores.topk(W, -1)
        _, ri = scores.topk(R, -1)
        m.step(keys[t], keys[t], vals[t], query_indices=ri, key_indices=wi, alpha=1.0, beta=betas[t].item())
    scores = k_star @ P
    _, ri = scores.topk(R, -1)
    out = m.step(k_star, k_star, torch.zeros(8), query_indices=ri, key_indices=ri, alpha=1.0, beta=0.0)
    return (out - v_star).norm().item()


def run_hola(D, cap, seed, cache_on=True):
    d_k = 64
    k_star, v_star, keys, vals, betas = build(seed, D, d_k)
    alpha = 0.95  
    m = HippocampusLinearAttention(d_k, 8, cap)
    for t in range(len(keys)):
        m.step(keys[t], keys[t], vals[t], alpha=alpha, beta=betas[t].item(),
               cache_gate=1.0 if cache_on else 0.0)
    out = m.step(k_star, k_star, torch.zeros(8), alpha=alpha, beta=0.0,
                 cache_gate=1.0 if cache_on else 0.0)
    return (out - v_star).norm().item()


print("SDM: N slots, W=R=8, d_k=8.  HOLA: d_k=64, surprise cache.")
print(f"{'gap':>5} | {'HOLA state':>12} | {'HOLA cache':>12} | {'SDM N=128':>11} | {'SDM N=1024':>12}")
print("-" * 58)
for D in distances:
    hs = sum(run_hola(D, 32, s, cache_on=False) for s in range(n_seeds)) / n_seeds
    hc = sum(run_hola(D, 32, s, cache_on=True) for s in range(n_seeds)) / n_seeds
    s128 = sum(run_sdm(D, 128, s) for s in range(n_seeds)) / n_seeds
    s1024 = sum(run_sdm(D, 1024, s) for s in range(n_seeds)) / n_seeds
    print(f"{D:>5} | {hs:>12.3f} | {hc:>12.3f} | {s128:>11.3f} | {s1024:>12.3f}")

print()
print("|v_star| ~ 13.7; errors near 13 mean the item was lost.")
