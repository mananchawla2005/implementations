import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.inference import (
    SoftmaxAttention,
    AdditiveLinear,
    GatedAdditive,
    DeltaNet,
    GatedDeltaNet,
)

nA = 20
nB = 20
d_k = 16
d_v = 16

torch.manual_seed(42)
keys_A = torch.randn(nA, d_k)
keys_A = keys_A / keys_A.norm(dim=1, keepdim=True)
vals_A = torch.randn(nA, d_v)

keys_B = torch.randn(nB, d_k)
keys_B = keys_B / keys_B.norm(dim=1, keepdim=True)
vals_B = torch.randn(nB, d_v)

model_classes = [
    ("Softmax", SoftmaxAttention),
    ("AdditiveLinear", AdditiveLinear),
    ("GatedAdditive", GatedAdditive),
    ("DeltaNet", DeltaNet),
    ("GatedDeltaNet", GatedDeltaNet),
]

results = {}

for name, cls in model_classes:
    model = cls(d_k, d_v)
    for i in range(nA):
        model.step(keys_A[i], keys_A[i], vals_A[i], alpha=1.0, beta=1.0)

    for i in range(nB):
        alpha = 0.1 if "Gated" in name else 1.0
        model.step(keys_B[i], keys_B[i], vals_B[i], alpha=alpha, beta=1.0)

    recall_B = 0.0
    for i in range(nB):
        out = model.step(keys_B[i], keys_B[i], torch.zeros(d_v), alpha=1.0, beta=0.0)
        recall_B += (out - vals_B[i]).pow(2).sum().item()
    recall_B = recall_B / nB

    ghosting = 0.0
    for i in range(nA):
        out = model.step(keys_A[i], keys_A[i], torch.zeros(d_v), alpha=1.0, beta=0.0)
        ghosting += out.norm().item()
    ghosting = ghosting / nA

    cross = 0.0
    for i in range(nA):
        for j in range(nB):
            out_A = model.step(keys_A[i], keys_A[i], torch.zeros(d_v), alpha=1.0, beta=0.0)
            out_B = model.step(keys_B[j], keys_B[j], torch.zeros(d_v), alpha=1.0, beta=0.0)
            cross += (out_A - out_B).norm().item()
    cross = cross / (nA * nB)

    results[name] = (recall_B, ghosting, cross)

print(f"{'Model':>16} | {'Recall B MSE':>14} | {'Ghosting':>10} | {'Cross-dist':>12}")
print("-" * 60)
for name, cls in model_classes:
    rB, g, c = results[name]
    print(f"{name:>16} | {rB:>13.6f}  | {g:>9.4f}  | {c:>11.4f}")
