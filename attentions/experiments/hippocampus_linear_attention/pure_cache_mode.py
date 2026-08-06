import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.hippocampus_linear_attention import hola_attention_chunked

d_k = 8
d_v = 6
T = 32
chunk_size = 8
capacity = T  # store every token
n_seeds = 50


def rms_norm(x, gamma):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8) * gamma


def run_seed(seed):
    g = torch.Generator().manual_seed(seed)
    raw_q = torch.randn(T, d_k, generator=g)
    raw_k = torch.randn(T, d_k, generator=g)
    values = torch.randn(T, d_v, generator=g)
    gamma_q = 0.5 + 0.5 * torch.rand(1, generator=g).item()
    gamma_k = 0.5 + 0.5 * torch.rand(1, generator=g).item()
    cache_q = rms_norm(raw_q, gamma_q)
    cache_k = rms_norm(raw_k, gamma_k)

    betas = torch.zeros(T)         
    lambdas = torch.ones(T)         
    alphas = torch.ones(T, d_k)     

    out = hola_attention_chunked(
        queries=raw_q,
        keys=raw_k,
        cache_queries=cache_q,
        cache_keys=cache_k,
        sink_logit=torch.tensor(0.0),
        values=values,
        betas=betas,
        alphas=alphas,
        cache_gates=lambdas,
        capacity=capacity,
        chunk_size=chunk_size,
    )

    logits = cache_q @ cache_k.transpose(0, 1) / (d_k ** 0.5)  # [T, T]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
    all_logits = torch.cat([logits, torch.zeros(T, 1)], dim=-1)  # sink
    weights = torch.softmax(all_logits, dim=-1)
    ref = weights[:, :-1] @ values

    return out, ref


max_overall = 0.0
print(f"{'seed':>5} | {'max |HOLA - causal cache|':>24}")
print("-" * 36)
for seed in range(n_seeds):
    out, ref = run_seed(seed)
    diff = (out - ref).abs().max().item()
    max_overall = max(max_overall, diff)
    if seed % 10 == 0:
        print(f"{seed:>5} | {diff:>24.3e}")

print("-" * 36)
print(f"max diff over {n_seeds} seeds: {max_overall:.3e}")

assert max_overall < 1e-5, "pure cache mode did not match explicit causal cache attention"