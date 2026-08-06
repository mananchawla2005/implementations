import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.hippocampus_linear_attention import hola_attention_chunked
from algorithms.inference import KimiDeltaAttention

d_k = 8
d_v = 6
T = 32
chunk_size = 8
capacity = 8
n_seeds = 50

print(f"{'seed':>5} | {'max |HOLA - KDA|':>18}")
print("-" * 32)
max_overall = 0.0

for seed in range(n_seeds):
    g = torch.Generator().manual_seed(seed)
    raw_q = torch.randn(T, d_k, generator=g)
    raw_k = torch.randn(T, d_k, generator=g)
    q = F.normalize(raw_q, dim=-1)
    k = F.normalize(raw_k, dim=-1)
    v = torch.randn(T, d_v, generator=g)
    betas = (torch.rand(T, generator=g) * 0.9 + 0.1)
    alphas = torch.rand(T, d_k, generator=g) * 0.9 + 0.1

    lambdas = torch.zeros(T)  

    out = hola_attention_chunked(
        queries=q,
        keys=k,
        cache_queries=q,
        cache_keys=k,
        sink_logit=torch.tensor(0.0),
        values=v,
        betas=betas,
        alphas=alphas,
        cache_gates=lambdas,
        capacity=capacity,
        chunk_size=chunk_size,
    )

    kda = KimiDeltaAttention(d_k, d_v)
    ref = []
    for t in range(T):
        o = kda.step(q[t], k[t], v[t], alpha=alphas[t], beta=betas[t].item())
        ref.append(o)
    ref = torch.stack(ref)

    diff = (out - ref).abs().max().item()
    max_overall = max(max_overall, diff)
    if seed % 10 == 0:
        print(f"{seed:>5} | {diff:>18.3e}")

print("-" * 32)
print(f"max diff over {n_seeds} seeds: {max_overall:.3e}")

assert max_overall < 1e-5, "HOLA(lambda=0) did not recover Kimi Delta Attention"