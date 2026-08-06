import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 64
d_v = 8
N_pred, N_fact, N_noise = 60, 8, 100
budgets = [1, 2, 4, 8, 16, 32, 64]
n_seeds = 5
torch.manual_seed(0)

ALPHA = 0.99
EXACT_THRESH = 1.0


def rms_norm(x):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8)


def build_sequence(seed):
    g = torch.Generator().manual_seed(seed)
    pred_keys = F.normalize(torch.randn(4, d_k, generator=g), dim=-1)  # repeated keys
    pred = [
        (pred_keys[i % 4], torch.randn(d_v, generator=g) * 0.1, 0.02, "pred")
        for i in range(N_pred)
    ]
    fact_keys = F.normalize(torch.randn(N_fact, d_k, generator=g), dim=-1)
    facts = [
        (fact_keys[i], torch.randn(d_v, generator=g) * 5.0, 1.0, "fact")
        for i in range(N_fact)
    ]
    noise = [
        (F.normalize(torch.randn(d_k, generator=g), dim=-1),
         torch.randn(d_v, generator=g) * 0.8, 0.1, "noise")
        for _ in range(N_noise)
    ]
    seq = pred[:20]
    for f in facts:
        seq.append(f)
        seq += noise[:10]
        noise = noise[10:]
    seq += pred[20:] + noise
    return seq, facts


def simulate(w, seed):
    seq, facts = build_sequence(seed)
    S = torch.zeros(d_k, d_v)
    cache_k, cache_v, cache_s = [], [], []  

    for k, v, b, tag in seq:
        e = v - S.T @ k
        S = ALPHA * S + b * torch.outer(k, e)
        score = b * e.norm().item()  
        cache_k.append(k)
        cache_v.append(v)
        cache_s.append((score, tag))
        if len(cache_s) > w:  
            idx = min(range(len(cache_s)), key=lambda i: cache_s[i][0])
            del cache_k[idx], cache_v[idx], cache_s[idx]

    fact_kept = sum(1 for (s, t) in cache_s if t == "fact")
    pred_kept = sum(1 for (s, t) in cache_s if t == "pred")
    noise_kept = sum(1 for (s, t) in cache_s if t == "noise")

    CK = rms_norm(torch.stack(cache_k))
    CV = torch.stack(cache_v)
    exact = 0
    for i in range(N_fact):
        q = facts[i][0]
        lg = rms_norm(q) @ CK.T / (d_k ** 0.5)
        wts = torch.softmax(lg, -1)
        ans = wts @ CV
        if (ans - facts[i][1]).norm().item() < EXACT_THRESH:
            exact += 1

    queried_retained = sum(1 for (s, t) in cache_s if t == "fact")
    precision = queried_retained / w            # retained tokens later queried / w
    recall = queried_retained / N_fact          # queried facts retained / queried facts

    return {
        "exact": exact,
        "fact_kept": fact_kept,
        "pred_kept": pred_kept,
        "noise_kept": noise_kept,
        "precision": precision,
        "recall": recall,
    }


print(f"{'w':>3} | {'exact':>7} | {'facts kept':>11} | {'pred kept':>10} | {'noise kept':>11} | {'precision':>10} | {'recall':>7}")
print("-" * 76)

rows = []
for w in budgets:
    agg = {k: 0.0 for k in ["exact", "fact_kept", "pred_kept", "noise_kept", "precision", "recall"]}
    for s in range(n_seeds):
        r = simulate(w, s)
        for k in agg:
            agg[k] += r[k]
    for k in agg:
        agg[k] /= n_seeds
    rows.append(agg)
    print(
        f"{w:>3} | {agg['exact']:>5.0f}/{N_fact} | {agg['fact_kept']:>6.1f}/{N_fact} | "
        f"{agg['pred_kept']:>10.1f} | {agg['noise_kept']:>11.1f} | "
        f"{agg['precision']:>10.2f} | {agg['recall']:>6.2f}"
    )

assert all(r["pred_kept"] < 1 for r in rows), "predictable tokens should not be cached"
assert rows[-1]["fact_kept"] >= N_fact - 0.5, "large budget should keep all facts"
assert rows[-1]["recall"] >= 0.95, "large budget should recall all queried facts"
