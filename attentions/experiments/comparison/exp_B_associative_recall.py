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
    KimiDeltaAttention,
)

N_values = [2, 4, 8, 16, 32, 64]
d_k_values = [4, 8, 16, 32]
n_seeds = 10

model_classes = [
    ("Softmax", SoftmaxAttention),
    ("AdditiveLinear", AdditiveLinear),
    ("GatedAdditive", GatedAdditive),
    ("DeltaNet", DeltaNet),
    ("GatedDeltaNet", GatedDeltaNet),
    ("KimiDelta", KimiDeltaAttention),
]

results = {}

for d_k in d_k_values:
    d_v = d_k
    for name, cls in model_classes:
        key_vals = []
        for N in N_values:
            seed_mses = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                keys = torch.randn(N, d_k)
                keys = keys / keys.norm(dim=1, keepdim=True)
                values = torch.randn(N, d_v)

                model = cls(d_k, d_v)
                for i in range(N):
                    if "Gated" in name:
                        model.step(keys[i], keys[i], values[i], alpha=1.0)
                    elif "Delta" in name:
                        model.step(keys[i], keys[i], values[i], beta=1.0)
                    else:
                        model.step(keys[i], keys[i], values[i])

                total_mse = 0.0
                for i in range(N):
                    out = model.step(keys[i], keys[i], torch.zeros(d_v), alpha=1.0, beta=0.0)
                    total_mse += (out - values[i]).pow(2).sum().item()

                seed_mses.append(total_mse / N)
            key_vals.append(sum(seed_mses) / len(seed_mses))
        results[(name, d_k)] = key_vals

header = f"{'Model':>16}" + "".join(f"{'N='+str(N):>8}" for N in N_values)
print(header)
print("-" * len(header))

for name, cls in model_classes:
    for d_k in d_k_values:
        row = f"{name} d={d_k:>2}"
        for mse in results[(name, d_k)]:
            row += f"{mse:>8.4f}"
        print(row)
    print()
