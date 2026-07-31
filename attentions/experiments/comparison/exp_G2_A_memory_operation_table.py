import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.inference import KimiDeltaAttention, GatedDeltaAttention2

d_k = 4
d_v = 2

torch.manual_seed(0)
k = torch.randn(d_k)
k = k / k.norm()

v1 = torch.tensor([1.0, 0.0])
v2 = torch.tensor([0.0, 1.0])
v3 = torch.tensor([0.5, -1.0])


ops = [
    ("REPLACE", v1, 1.0, 1.0, torch.tensor([1.0, 0.0]), 1.0),
    ("INSERT ", v2, 0.0, 1.0, torch.tensor([1.0, 1.0]), 1.0),
    ("DELETE ", v3, 1.0, 0.0, torch.tensor([0.0, 0.0]), 1.0),
    ("IGNORE ", v3, 0.0, 0.0, torch.tensor([0.0, 0.0]), 0.0),
]


def fmt(v):
    return f"[{v[0]:+7.2f}, {v[1]:+7.2f}]"


def read(model):
    return model.step(k, k, torch.zeros(d_v), alpha=1.0, beta=0.0)


gdn2 = GatedDeltaAttention2(d_k, d_v)
kda = KimiDeltaAttention(d_k, d_v)

print(f"{'op':>8} | {'b':>3} {'w':>3} | {'ideal':>18} | {'GDN2':>18} {'ok':>3} | {'KDA':>18} {'ok':>3}")
print("-" * 78)
for name, v, b, w, ideal, kda_beta in ops:
    gdn2.step(k, k, v, alpha=1.0, erase_gate=b, write_gate=w)
    g = read(gdn2)
    kda.step(k, k, v, alpha=1.0, beta=kda_beta)
    kd = read(kda)
    gok = (g - ideal).abs().max().item() < 1e-3
    kok = (kd - ideal).abs().max().item() < 1e-3
    print(
        f"{name:>8} | {b:>3.0f} {w:>3.0f} | {fmt(ideal)} | {fmt(g)} "
        f"{'YES' if gok else 'no':>3} | {fmt(kd)} {'YES' if kok else 'no':>3}"
    )
