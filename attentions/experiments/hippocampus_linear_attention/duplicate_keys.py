import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.hippocampus_linear_attention import hola_attention_chunked

d_k = 64
d_v = 4
chunk_size = 8
T = 16  

torch.manual_seed(0)
g = torch.Generator().manual_seed(0)

k_dup = F.normalize(torch.randn(d_k, generator=g), dim=-1)
v1 = torch.randn(d_v, generator=g)
v2 = torch.randn(d_v, generator=g)


def rms_norm(x, gamma):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8) * gamma


def run(ka, kb, kq):
    keys = torch.randn(T, d_k, generator=g)
    keys[0] = ka
    keys[1] = kb
    keys[8] = kq
    keys = F.normalize(keys, dim=-1)
    queries = keys.clone()
    queries[8] = kq

    values = torch.zeros(T, d_v)
    values[0] = v1
    values[1] = v2

    betas = torch.zeros(T)
    betas[0:2] = 1.0
    alphas = torch.ones(T, d_k)

    out = hola_attention_chunked(
        queries=queries,
        keys=keys,
        cache_queries=rms_norm(queries, 1.0),
        cache_keys=rms_norm(keys, 1.0),
        sink_logit=torch.tensor(0.0),
        values=values,
        betas=betas,
        alphas=alphas,
        cache_gates=torch.ones(T),
        capacity=16,
        chunk_size=chunk_size,
    )
    return out[8]


def cache_retrieve(stored_k, stored_v, qk):
    logits = qk @ stored_k.T / (d_k ** 0.5)
    logits = torch.cat([logits, torch.tensor([-50.0])])  
    w = torch.softmax(logits, -1)
    return w[:-1] @ stored_v, w[:-1]


print("DUPLICATE keys: both items share key k_dup")
w_dup, wts = cache_retrieve(torch.stack([k_dup, k_dup]), torch.stack([v1, v2]), k_dup)
print("  attention weights:", wts.tolist(), "(equal, key alone cannot distinguish)")
err1 = (w_dup - v1).norm().item()
err2 = (w_dup - v2).norm().item()
err_blend = (w_dup - (v1 + v2) / 2).norm().item()
print(f"  retrieval error to v1: {err1:.3f}")
print(f"  retrieval error to v2: {err2:.3f}")
print(f"  retrieval error to blend (v1+v2)/2: {err_blend:.3f}")

dup_retr = run(k_dup, k_dup, k_dup)
print(f"  [HOLA end-to-end] err to v1: {(dup_retr-v1).norm().item():.3f}, "
      f"err to v2: {(dup_retr-v2).norm().item():.3f}")

assert abs(wts[0].item() - wts[1].item()) < 1e-6, "duplicate keys must get equal weight"
assert err_blend < err1 and err_blend < err2, "retrieval is a blend, not either exact item"
print("\nidentical keys blend: neither exact item is recoverable")

print("\nDISTINCT (orthogonal) keys: control ===")
k_a = F.normalize(torch.randn(d_k, generator=g), dim=-1)
k_b = F.normalize(torch.randn(d_k, generator=g), dim=-1)
k_b = k_b - (k_b @ k_a) * k_a
k_b = F.normalize(k_b, dim=-1)

w_a, wa_wts = cache_retrieve(torch.stack([k_a, k_b]), torch.stack([v1, v2]), k_a)
w_b, wb_wts = cache_retrieve(torch.stack([k_a, k_b]), torch.stack([v1, v2]), k_b)
err_a_to_v1 = (w_a - v1).norm().item()
err_a_to_v2 = (w_a - v2).norm().item()
err_b_to_v2 = (w_b - v2).norm().item()
err_b_to_v1 = (w_b - v1).norm().item()
print(f"  query k_a weights: {wa_wts.tolist()}  (right item dominates)")
print(f"  query k_a error to v1: {err_a_to_v1:.3f}  vs  to v2: {err_a_to_v2:.3f}")
print(f"  query k_b error to v2: {err_b_to_v2:.3f}  vs  to v1: {err_b_to_v1:.3f}")

assert wa_wts[0].item() > wa_wts[1].item(), "query k_a should favor v1's key"
assert wb_wts[1].item() > wb_wts[0].item(), "query k_b should favor v2's key"
assert err_a_to_v1 < err_a_to_v2, "query k_a should recover v1 better than v2"
assert err_b_to_v2 < err_b_to_v1, "query k_b should recover v2 better than v1"
