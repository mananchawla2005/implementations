import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

W = R = 8
d_v = 16
n_seeds = 10

N_values = [16, 64, 256, 1024, 4096]


def simulate(N, M, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(M, N, generator=g)         
    write_weights, write_idx = scores.topk(W, dim=-1)
    write_weights = F.softmax(write_weights, -1)
    values = torch.randn(M, d_v, generator=g)

    T = 2 * M  # M write steps, then M read steps
    keys = torch.cat([write_weights, torch.zeros(M, W)])
    queries = torch.cat([torch.zeros(M, W), write_weights])
    key_indices = torch.cat([write_idx, write_idx])
    query_indices = torch.cat([write_idx, write_idx])
    all_values = torch.cat([values, torch.zeros(M, d_v)])
    alphas = torch.ones(T)
    betas = torch.ones(T)
    betas[M:] = 0.0  # read steps write nothing

    outputs, memories = sparse_delta_memory_recurrent(
        queries, query_indices, keys, key_indices, all_values, alphas, betas, N
    )

    retr = outputs[M:]
    mse = ((retr - values) ** 2).mean().item()

    final = memories[-1]
    slot_norms = final.norm(dim=1)
    used = (slot_norms > 1e-8).sum().item()
    frac_unique = used / N
    avg_writes = (M * W) / max(used, 1)

    occupied = torch.zeros(N, dtype=torch.bool)
    collisions = 0
    total_checks = 0
    for t in range(M):
        for s in write_idx[t].tolist():
            if occupied[s]:
                collisions += 1
            occupied[s] = True
            total_checks += 1
    collision_rate = collisions / max(total_checks, 1)

    rel_err = ((retr - values) ** 2).sum(dim=1) / (values ** 2).sum(dim=1)
    accuracy = (rel_err < 0.50).float().mean().item()

    return mse, frac_unique, collision_rate, avg_writes, accuracy


def table(M_label, M_fn):
    print(f"\n{'N':>6} | {'recall MSE':>11} | {'unique':>8} | {'collision':>10} | {'writes/slot':>12} | {'acc<50%':>8}")
    print("-" * 68)
    for N in N_values:
        M = M_fn(N)
        ms, frac, col, aw, acc = [], [], [], [], []
        for seed in range(n_seeds):
            r = simulate(N, M, seed)
            ms.append(r[0]); frac.append(r[1]); col.append(r[2]); aw.append(r[3]); acc.append(r[4])
        print(
            f"{N:>6} | {sum(ms)/n_seeds:>11.4f} | {sum(frac)/n_seeds:>7.3f} | "
            f"{sum(col)/n_seeds:>10.3f} | {sum(aw)/n_seeds:>12.1f} | {sum(acc)/n_seeds:>7.3f}"
        )


print("fixed load: M = 64 associations ===")
table("M=64", lambda N: 64)

print("\nproportional load: M = N associations ===")
table("M=N", lambda N: N)
