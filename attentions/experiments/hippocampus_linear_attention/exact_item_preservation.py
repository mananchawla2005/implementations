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
n_read = 8
T = n_writes + n_read
chunk_size = 8
capacity = 8

torch.manual_seed(0)


def rms_norm(x, gamma):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8) * gamma


g = torch.Generator().manual_seed(0)
k_star = F.normalize(torch.randn(d_k, generator=g), dim=-1)
v_star = torch.randn(d_v, generator=g) * 5.0   # large norm -> top surprise

keys = torch.randn(T, d_k, generator=g)
keys[0] = k_star
keys[n_writes:] = torch.randn(n_read, d_k, generator=g)  # read steps -> distinct keys
keys = F.normalize(keys, dim=-1)

values = torch.zeros(T, d_v)
values[0] = v_star
values[1:n_writes] = torch.randn(n_writes - 1, d_v, generator=g)  # interfering writes

betas = torch.zeros(T)
betas[:n_writes] = 1.0
alphas = torch.ones(T, d_k)

# read steps query the target key but write nothing
queries = keys.clone()
queries[n_writes:] = k_star.unsqueeze(0).expand(n_read, d_k)

cache_queries = rms_norm(queries, 1.0)
cache_keys = rms_norm(keys, 1.0)


def run(cache_gates, capacity):
    out = hola_attention_chunked(
        queries=queries,
        keys=keys,
        cache_queries=cache_queries,
        cache_keys=cache_keys,
        sink_logit=torch.tensor(0.0),
        values=values,
        betas=betas,
        alphas=alphas,
        cache_gates=cache_gates,
        capacity=capacity,
        chunk_size=chunk_size,
    )
    return out[n_writes:] 


def err(out):
    return (out - v_star).norm(dim=1).mean().item()


state_only = run(torch.zeros(T), capacity)     
with_cache = run(torch.ones(T), capacity)    

err_state = err(state_only)
err_cache = err(with_cache)

print(f"|v_star| = {v_star.norm().item():.2f}")
print(f"state-only retrieval error : {err_state:6.3f}   (degraded)")
print(f"with-cache retrieval error : {err_cache:6.3f}   (preserved)")

print("\nerror while target stays selected, across capacities:")
errs = []
for cap in [2, 4, 8, 16, 32]:
    e = err(run(torch.ones(T), cap))
    errs.append(e)
    print(f"  capacity={cap:3d}: {e:.3f}")

assert err_cache < err_state / 3, "cache did not preserve the target"
assert err_state > v_star.norm().item() * 0.8, "state interference too weak"
assert max(errs) < err_cache * 1.2, "target was evicted at some capacity"