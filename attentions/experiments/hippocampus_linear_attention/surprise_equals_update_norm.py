import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.hippocampus_linear_attention import hola_attention_chunked

d_k = 16
d_v = 8
T = 16
chunk_size = 8
capacity = 4
n_seeds = 200


def rms_norm(x, gamma):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8) * gamma


def run_seed(seed, alpha, beta, cache_gates):
    g = torch.Generator().manual_seed(seed)
    raw_q = torch.randn(T, d_k, generator=g)
    raw_k = torch.randn(T, d_k, generator=g)
    values = torch.randn(T, d_v, generator=g)

    # state path: L2-normalized
    state_q = F.normalize(raw_q, dim=-1)
    state_k = F.normalize(raw_k, dim=-1)

    # cache path: RMS-normalized with separate scales
    gamma_q = 0.5 + 0.5 * torch.rand(1, generator=g).item()
    gamma_k = 0.5 + 0.5 * torch.rand(1, generator=g).item()
    cache_q = rms_norm(raw_q, gamma_q)
    cache_k = rms_norm(raw_k, gamma_k)

    out = hola_attention_chunked(
        queries=state_q,
        keys=state_k,
        cache_queries=cache_q,
        cache_keys=cache_k,
        sink_logit=torch.tensor(0.0),
        values=values,
        betas=beta,
        alphas=alpha,
        cache_gates=cache_gates,
        capacity=capacity,
        chunk_size=chunk_size,
    )

    S = torch.zeros(d_k, d_v)
    surprise_norms = []
    write_norms = []
    ref_out = []
    for t in range(T):
        decayed = alpha[t, :, None] * S
        pred = decayed.T @ state_k[t]
        e = values[t] - pred
        S_next = decayed + beta[t] * torch.outer(state_k[t], e)
        surprise_norms.append(e.norm().item())
        write_norms.append((S_next - decayed).norm("fro").item())
        ref_out.append(S_next.T @ state_q[t])
        S = S_next

    return out, torch.stack(ref_out), torch.tensor(surprise_norms), torch.tensor(write_norms)


max_diff = 0.0
max_surprise_diff = 0.0
all_ok = True
all_surprise_ok = True

for seed in range(n_seeds):
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(T, d_k, generator=g) * 0.9 + 0.1
    beta = torch.rand(T, generator=g) * 0.9 + 0.1
    cache_gates = torch.zeros(T) 

    out, ref, surprise, write_norm = run_seed(seed, alpha, beta, cache_gates)

    out_diff = (out - ref).abs().max().item()
    max_diff = max(max_diff, out_diff)

    sr_diff = (write_norm - beta * surprise).abs().max().item()
    max_surprise_diff = max(max_surprise_diff, sr_diff)
    if out_diff > 1e-3:
        all_ok = False
    if sr_diff > 1e-4:
        all_surprise_ok = False

print(f"max |hola_attention_chunked - recurrent| over {n_seeds} seeds: {max_diff:.3e}")
print(f"max |surprise norm - write magnitude|   over {n_seeds} seeds: {max_surprise_diff:.3e}")

assert all_ok, "chunked output diverged from recurrent reference"
assert all_surprise_ok, "HOLA-1 failed"