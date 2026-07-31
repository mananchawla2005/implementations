import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
from algorithms.inference import (
    SoftmaxAttention,
    AdditiveLinear,
    GatedAdditive,
    DeltaNet,
    GatedDeltaNet,
    KimiDeltaAttention,
)
from shared.plotting import plot_times

d_v = 2

k1 = torch.tensor([1.0, 0.0])
v1 = torch.randn(d_v)
v2 = torch.randn(d_v)
v1_new = torch.randn(d_v)

thetas = [i * math.pi / 40 for i in range(21)]
model_classes = [
    ("Softmax", SoftmaxAttention),
    ("AdditiveLinear", AdditiveLinear),
    ("GatedAdditive", GatedAdditive),
    ("DeltaNet", DeltaNet),
    ("GatedDeltaNet", GatedDeltaNet),
    ("KimiDelta", KimiDeltaAttention),
]

all_damage = {}

for name, cls in model_classes:
    damage = []
    for theta in thetas:
        k2 = torch.tensor([math.cos(theta), math.sin(theta)])
        torch.manual_seed(0)
        v1 = torch.randn(d_v)
        v2 = torch.randn(d_v)
        v1_new = torch.randn(d_v)

        model = cls(2, d_v)
        model.step(k1, k1, v1, alpha=1.0, beta=1.0)
        model.step(k2, k2, v2, alpha=1.0, beta=1.0)
        read_before = model.step(k2, k2, torch.zeros(d_v), alpha=1.0, beta=0.0)
        model.step(k1, k1, v1_new, alpha=1.0, beta=1.0)
        read_after = model.step(k2, k2, torch.zeros(d_v), alpha=1.0, beta=0.0)

        dmg = (read_after - read_before).norm().item()
        damage.append(dmg)
    all_damage[name] = damage

dots = [k1 @ torch.tensor([math.cos(t), math.sin(t)]) for t in thetas]

print(f"{'cos(theta)':>10} ", end="")
for name, _, in model_classes:
    print(f"{name:>16}", end="")
print()
for i, cos_theta in enumerate(dots):
    print(f"{cos_theta:>10.4f} ", end="")
    for name, _, in model_classes:
        print(f"{all_damage[name][i]:>15.6f} ", end="")
    print()

plot_times(
    dots,
    all_damage,
    xlabel=r"k1 . k2 = cos(theta)",
    ylabel="Damage to k2 retrieval",
    title="Interference vs Key Similarity",
)
