import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt

n_seeds = 200


def collision_rate(A, N, W, seed):
    g = torch.Generator().manual_seed(seed)
    used = torch.zeros(N, dtype=torch.bool)
    for _ in range(A):
        scores = torch.rand(N, generator=g)
        slots = scores.topk(W, dim=-1).indices
        used[slots] = True
    total = A * W
    unique = used.sum().item()
    return 1.0 - unique / total, unique


def mean_col(A, N, W):
    vals = [collision_rate(A, N, W, s)[0] for s in range(n_seeds)]
    return sum(vals) / n_seeds


N, W = 256, 8
print(f"1) collision rate vs associations  (N={N}, W={W})")
print(f"{'A':>6} | {'collision':>10}")
for A in [8, 16, 32, 64, 128, 256, 512, 1024]:
    print(f"{A:>6} | {mean_col(A, N, W):>10.3f}")

A, W = 64, 8
print(f"\n2) collision rate vs memory size  (A={A}, W={W})")
print(f"{'N':>7} | {'collision':>10}")
for N in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    print(f"{N:>7} | {mean_col(A, N, W):>10.3f}")

A, N = 64, 256
print(f"\n3) collision rate vs write budget  (A={A}, N={N})")
print(f"{'W':>3} | {'collision':>10}")
for W in [1, 2, 4, 8, 16, 32, 64]:
    print(f"{W:>3} | {mean_col(A, N, W):>10.3f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

As = [8, 16, 32, 64, 128, 256, 512, 1024]
axes[0].plot(As, [mean_col(a, 256, 8) for a in As], marker="o")
axes[0].set_xlabel("associations A")
axes[0].set_ylabel("collision rate")
axes[0].set_title("N=256, W=8")
axes[0].grid(True, linestyle="--", alpha=0.6)

Ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
axes[1].plot(Ns, [mean_col(64, n, 8) for n in Ns], marker="o")
axes[1].set_xlabel("memory size N")
axes[1].set_ylabel("collision rate")
axes[1].set_title("A=64, W=8")
axes[1].grid(True, linestyle="--", alpha=0.6)

Ws = [1, 2, 4, 8, 16, 32, 64]
axes[2].plot(Ws, [mean_col(64, 256, w) for w in Ws], marker="o")
axes[2].set_xlabel("write budget W")
axes[2].set_ylabel("collision rate")
axes[2].set_title("A=64, N=256")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
