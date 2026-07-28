import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from algorithms.inference import (
    SoftmaxAttention,
    AdditiveLinear,
    GatedAdditive,
    DeltaNet,
    GatedDeltaNet,
)
from shared.plotting import plot_times

d_k = 64
d_v = 64

seq_lens = [10, 50, 100, 200, 500, 1000]

model_classes = [
    ("Softmax", SoftmaxAttention),
    ("AdditiveLinear", AdditiveLinear),
    ("GatedAdditive", GatedAdditive),
    ("DeltaNet", DeltaNet),
    ("GatedDeltaNet", GatedDeltaNet),
]

all_times = {}
all_stored = {}

for name, cls in model_classes:
    runtimes = []
    stored = []
    for T in seq_lens:
        torch.manual_seed(0)
        keys = torch.randn(T, d_k)
        keys = keys / keys.norm(dim=1, keepdim=True)
        values = torch.randn(T, d_v)

        model = cls(d_k, d_v)
        start = time.perf_counter()
        for i in range(T):
            if "Gated" in name:
                model.step(keys[i], keys[i], values[i], alpha=1.0)
            elif "Delta" in name:
                model.step(keys[i], keys[i], values[i], beta=1.0)
            else:
                model.step(keys[i], keys[i], values[i])
        elapsed = time.perf_counter() - start
        runtimes.append(elapsed)

        if name == "Softmax":
            stored.append(T * (d_k + d_v))
        else:
            stored.append(d_k * d_v)
    all_times[name] = runtimes
    all_stored[name] = stored

print(f"{'Seq len':>8} ", end="")
for name, _ in model_classes:
    print(f"{name:>16}", end="")
print()
for idx, T in enumerate(seq_lens):
    print(f"{T:>8} ", end="")
    for name, _ in model_classes:
        print(f"{all_times[name][idx]:>15.6f}s ", end="")
    print()

print()
print(f"{'Stored scalars (T=1000)':>30}")
for name, _ in model_classes:
    print(f"{name:>16}: {all_stored[name][-1]:>8}")

plot_times(
    seq_lens,
    all_times,
    xlabel="Sequence length",
    ylabel="Total runtime (s)",
    title="Inference Time vs Sequence Length (d=64)",
)
