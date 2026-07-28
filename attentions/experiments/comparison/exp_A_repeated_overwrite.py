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

d_k = 4
d_v = 2

key = torch.ones(d_k) / (d_k ** 0.5)

v1 = torch.tensor([1.0, 0.0])
v2 = torch.tensor([0.0, 1.0])
v3 = torch.tensor([-1.0, 0.0])

models = {
    "Softmax": SoftmaxAttention(d_k, d_v),
    "AdditiveLinear": AdditiveLinear(d_k, d_v),
    "GatedAdditive": GatedAdditive(d_k, d_v),
    "DeltaNet": DeltaNet(d_k, d_v),
    "GatedDeltaNet": GatedDeltaNet(d_k, d_v),
}

values = [v1, v2, v3]

print(f"{'Model':>16} | {'MSE after v1':>14} | {'MSE after v2':>14} | {'MSE after v3':>14} | {'Behavior':>30}")
print("-" * 95)

for name, model in models.items():
    model.reset()
    mses = []
    for v in values:
        if "Gated" in name:
            out = model.step(key, key, v, alpha=0.5)
        elif "Delta" in name:
            out = model.step(key, key, v, beta=1.0)
        else:
            out = model.step(key, key, v)
        mse = (out - v).pow(2).sum().item()
        mses.append(mse)

    desc = {
        "Softmax": "mixture of all past values",
        "AdditiveLinear": "accumulates",
        "GatedAdditive": "decayed mixture",
        "DeltaNet": "exact replacement",
        "GatedDeltaNet": "replacement + decay",
    }[name]
    print(f"{name:>16} | {mses[0]:>13.6f}  | {mses[1]:>13.6f}  | {mses[2]:>13.6f}  | {desc:>30}")
