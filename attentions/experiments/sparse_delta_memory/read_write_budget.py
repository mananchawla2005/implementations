import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import torch.nn.functional as F
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

A = 64       
N = 256      
d_v = 16
d_k = 32     
budgets = [1, 2, 4, 8, 16, 32]
n_seeds = 5

FOUR_BYTES = 4


def simulate(W, R, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(A, N, generator=g)
    values = torch.randn(A, d_v, generator=g)

    ww, wix = scores.topk(W, -1)
    ww = F.softmax(ww, -1)
    qq, qix = scores.topk(R, -1)
    qq = F.softmax(qq, -1)

    # zero-pad to d_k
    keys = torch.cat([ww, torch.zeros(A, d_k - W)], -1)
    queries = torch.cat([qq, torch.zeros(A, d_k - R)], -1)
    key_indices = torch.cat([wix, torch.zeros(A, d_k - W, dtype=torch.long)], -1)
    query_indices = torch.cat([qix, torch.zeros(A, d_k - R, dtype=torch.long)], -1)

    T = 2 * A
    all_keys = torch.cat([keys, torch.zeros(A, d_k)])
    all_q = torch.cat([torch.zeros(A, d_k), queries])
    all_kidx = torch.cat([key_indices, key_indices])
    all_qidx = torch.cat([query_indices, query_indices])
    all_v = torch.cat([values, torch.zeros(A, d_v)])
    alphas = torch.ones(T)
    betas = torch.ones(T)
    betas[A:] = 0.0

    t0 = time.perf_counter()
    outputs, memories = sparse_delta_memory_recurrent(
        all_q, all_qidx, all_keys, all_kidx, all_v, alphas, betas, N
    )
    dt = time.perf_counter() - t0

    retr = outputs[A:]
    mse = ((retr - values) ** 2).mean().item()
    rel_err = ((retr - values) ** 2).sum(1) / ((values ** 2).sum(1) + 1e-9)
    acc = (rel_err < 0.50).float().mean().item()

    # memory traffic (bytes): write = read old W + write new W values per
    # write step, read = fetch R values per read step.
    traffic = (A * (W + W + R)) * d_v * FOUR_BYTES

    touched = memories[-1].norm(dim=1) > 1e-8
    eff_slots = touched.sum().item()

    return mse, acc, dt, traffic, eff_slots


print(f"A={A} associations, N={N} slots, d_v={d_v}")
print(f"\n{'W':>3} {'R':>3} | {'recall MSE':>11} | {'acc<50%':>8} | {'runtime ms':>11} | {'traffic KB':>10} | {'eff slots':>10}")
print("-" * 82)

for W in budgets:
    for R in budgets:
        m, a, t, tr, e = [sum(x) / n_seeds for x in zip(*[simulate(W, R, s) for s in range(n_seeds)])]
        print(
            f"{W:>3} {R:>3} | {m:>11.4f} | {a:>7.3f} | {t*1e3:>10.2f} | "
            f"{tr/1e3:>10.1f} | {e:>10.0f}"
        )
