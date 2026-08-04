import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.inference import GatedDeltaNet
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent

d_k_dense = 64         
W = 32
R = 32                  # W + R = 64 = d_k_dense  (matched update cost)
d_v = 16
N_values = [128, 256, 1024, 4096]
M_values = [16, 64, 128, 256, 512, 1024]
n_seeds = 5

print(f"Matched update cost: dense touches {d_k_dense} rows/step; "
      f"sparse touches W+R = {W}+{R} = {W + R} slots/step")
print(f"Value dimension d_v = {d_v}\n")

Ms = M_values


def dense_recall(M, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(M, d_k_dense, generator=g)
    keys = keys / keys.norm(dim=1, keepdim=True)
    vals = torch.randn(M, d_v, generator=g)
    gdn = GatedDeltaNet(d_k_dense, d_v)
    for i in range(M):
        gdn.step(keys[i], keys[i], vals[i], alpha=1.0, beta=1.0)
    retr = torch.stack([
        gdn.step(keys[i], keys[i], torch.zeros(d_v), alpha=1.0, beta=0.0)
        for i in range(M)
    ])
    return ((retr - vals) ** 2).mean().item()


def sparse_recall(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(M, N, generator=g)
    ww, wix = scores.topk(W, -1)
    ww = F.softmax(ww, -1)
    qq, qix = scores.topk(R, -1)
    qq = F.softmax(qq, -1)
    vals = torch.randn(M, d_v, generator=g)
    T = 2 * M
    keys = torch.cat([ww, torch.zeros(M, W)])
    queries = torch.cat([torch.zeros(M, W), qq])
    key_indices = torch.cat([wix, wix])
    query_indices = torch.cat([qix, qix])
    all_v = torch.cat([vals, torch.zeros(M, d_v)])
    alphas = torch.ones(T)
    betas = torch.ones(T)
    betas[M:] = 0.0  # read steps write nothing
    outputs, _ = sparse_delta_memory_recurrent(
        queries, query_indices, keys, key_indices, all_v, alphas, betas, N
    )
    retr = outputs[M:]
    return ((retr - vals) ** 2).mean().item()


print(f"{'pairs M':>7} | {'dense':>10} |" + "".join(f"{f'N={N}':>12}" for N in N_values))
print("-" * (20 + 12 * len(N_values)))

dense_row = []
for M in Ms:
    dense_row.append(sum(dense_recall(M, s) for s in range(n_seeds)) / n_seeds)

sparse_rows = {}
for N in N_values:
    sparse_rows[N] = []
    for M in Ms:
        sparse_rows[N].append(sum(sparse_recall(M, N, s) for s in range(n_seeds)) / n_seeds)

for i, M in enumerate(Ms):
    line = f"{M:>7} | {dense_row[i]:>9.3f} |"
    for N in N_values:
        line += f"{sparse_rows[N][i]:>12.3f}"
    print(line)
