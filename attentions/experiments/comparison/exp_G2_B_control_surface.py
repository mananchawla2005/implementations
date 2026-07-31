import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt
from algorithms.inference import KimiDeltaAttention, GatedDeltaAttention2

d_k = 4
d_v = 2

torch.manual_seed(0)
k = torch.randn(d_k)
k = k / k.norm()

v_old = torch.tensor([10.0, 20.0])
v_new = torch.tensor([100.0, -50.0])

grid = [0.0, 0.25, 0.5, 1.0]


def fmt(v):
    return f"[{v[0]:+7.2f}, {v[1]:+7.2f}]"


def read(model):
    return model.step(k, k, torch.zeros(d_v), alpha=1.0, beta=0.0)


def store(model):
    model.step(k, k, v_old, alpha=1.0, erase_gate=1.0, write_gate=1.0)


surf = torch.zeros(len(grid), len(grid), d_v)  # [b, w, dim]
for i, b in enumerate(grid):
    for j, w in enumerate(grid):
        m = GatedDeltaAttention2(d_k, d_v)
        store(m)
        m.step(k, k, v_new, alpha=1.0, erase_gate=b, write_gate=w)
        surf[i, j] = read(m)

hdr = "w \\ b"
print("GDN2 retrieval after uncertain update (rows = write gate w):")
print(f"{hdr:>8}" + "".join(f"{b:>11.2f}" for b in grid))
for j, w in enumerate(grid):
    dim0 = "".join(f"{surf[i, j, 0]:>11.2f}" for i in range(len(grid)))
    print(f"{'':>8}{dim0}")
print(f"{hdr:>8}" + "".join(f"{b:>11.2f}" for b in grid))
for j, w in enumerate(grid):
    dim1 = "".join(f"{surf[i, j, 1]:>11.2f}" for i in range(len(grid)))
    print(f"{'':>8}{dim1}")

print("\nKDA diagonal (b = w = beta):")
for b in grid:
    m = KimiDeltaAttention(d_k, d_v)
    store(m)
    m.step(k, k, v_new, alpha=1.0, beta=b)
    print(f"  beta={b:>4.2f} -> {fmt(read(m))}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for dim in range(d_v):
    ax = axes[dim]
    im = ax.imshow(surf[:, :, dim].T, origin="lower", cmap="RdBu_r")
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([f"{g:.2f}" for g in grid])
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([f"{g:.2f}" for g in grid])
    ax.set_xlabel("erase gate b")
    ax.set_ylabel("write gate w")
    ax.set_title(f"retrieved dim {dim}")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
