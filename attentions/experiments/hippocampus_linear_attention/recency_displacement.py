import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.hippocampus_linear_attention import hola_attention_chunked

d_k = 64
d_v = 8
n_writes = 240
T = n_writes
chunk_size = 8

torch.manual_seed(0)

g = torch.Generator().manual_seed(0)
k_star = F.normalize(torch.randn(d_k, generator=g), dim=-1)
v_star = torch.randn(d_v, generator=g) * 5.0  

keys = torch.randn(T, d_k, generator=g)
keys[0] = k_star
keys = F.normalize(keys, dim=-1)

values = torch.zeros(T, d_v)
values[0] = v_star
values[1:] = torch.randn(T - 1, d_v, generator=g) * 0.3  

S = torch.zeros(d_k, d_v)
surprise = []
for t in range(T):
    k = F.normalize(keys[t].unsqueeze(0), 2, 1).squeeze(0)
    e = values[t] - S.T @ k
    surprise.append(e.norm().item())
    S = S + torch.outer(k, e)
surprise = torch.tensor(surprise)

print(f"target surprise: {surprise[0]:.2f}, median recent surprise: {surprise[1:].median():.2f}")

print(f"\n{'w':>4} | {'recency keeps target':>21} | {'surprise keeps target':>22}")
for w in [8, 16, 32, 64]:
    recency_keeps = 0 in set(range(T - w, T))                     
    surprise_keeps = 0 in set(surprise.topk(w).indices.tolist())   
    print(f"{w:>4} | {str(recency_keeps):>21} | {str(surprise_keeps):>22}")

print("\nretrieval error of target v_star at the end of the sequence:")
for w in [8, 16]:
    recency_errs, surprise_errs = [], []
    for _ in range(5):
        idx = torch.arange(T - w, T)
        cache_k = keys[idx]
        cache_v = values[idx]
        logits = k_star @ cache_k.T / (d_k ** 0.5)
        wgt = torch.softmax(logits, -1)
        recency_retr = wgt @ cache_v
        recency_errs.append((recency_retr - v_star).norm().item())

        n_read = 8
        Tt = n_writes + n_read
        assert Tt % chunk_size == 0
        full_keys = torch.cat([keys, torch.randn(n_read, d_k, generator=g)])
        full_keys = F.normalize(full_keys, dim=-1)
        full_vals = torch.cat([values, torch.zeros(n_read, d_v)])
        betas = torch.zeros(Tt); betas[:n_writes] = 1.0
        alphas = torch.ones(Tt, d_k)
        q = full_keys.clone()
        q[n_writes:] = k_star.unsqueeze(0).expand(n_read, d_k)

        def rms_norm(x):
            return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8)

        out = hola_attention_chunked(
            queries=q,
            keys=full_keys,
            cache_queries=rms_norm(q),
            cache_keys=rms_norm(full_keys),
            sink_logit=torch.tensor(0.0),
            values=full_vals,
            betas=betas,
            alphas=alphas,
            cache_gates=torch.ones(Tt),
            capacity=w,
            chunk_size=chunk_size,
        )
        surprise_errs.append((out[n_writes:] - v_star).norm(dim=1).mean().item())

    print(f"  w={w:>2}: recency cache error = {sum(recency_errs)/5:6.3f}   "
          f"surprise cache error = {sum(surprise_errs)/5:6.3f}")
    assert sum(recency_errs) / 5 > sum(surprise_errs) / 5, "surprise cache should retain better"

assert surprise[0] > surprise[1:].max() * 2, "target should dominate the surprise score"