import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.inference import KimiDeltaAttention, GatedDeltaAttention2

# Gating the error term too (error = w * (v - prediction)) gives true field-level editing.

d_k = 4
d_v = 4

torch.manual_seed(0)
k = torch.randn(d_k)
k = k / k.norm()

fields = ["identity", "location", "confidence", "timestamp"]
v_old = torch.tensor([1.0, 10.0, 0.9, 100.0])
v_new = torch.tensor([0.0, 50.0, 0.1, 200.0])
w = torch.tensor([0.0, 1.0, 0.0, 1.0])


def fmt(v):
    return "[" + ", ".join(f"{x:8.2f}" for x in v.tolist()) + "]"


def read(model):
    return model.step(k, k, torch.zeros(d_v), alpha=1.0, beta=0.0)


def run_gdn2(b):
    m = GatedDeltaAttention2(d_k, d_v)
    m.step(k, k, v_old, alpha=1.0, erase_gate=1.0, write_gate=1.0)
    m.step(k, k, v_new, alpha=1.0, erase_gate=b, write_gate=w)
    return read(m)


# masked-error variant
def run_masked():
    m = GatedDeltaAttention2(d_k, d_v)
    m.step(k, k, v_old, alpha=1.0, erase_gate=1.0, write_gate=1.0)
    pred = m.state.T @ k
    m.state = m.state + torch.outer(k, w * (v_new - pred))
    return read(m)


def run_kda():
    m = KimiDeltaAttention(d_k, d_v)
    m.step(k, k, v_old, alpha=1.0, beta=1.0)
    m.step(k, k, v_new, alpha=1.0, beta=1.0)
    return read(m)


print(f"{'method':>26} | {'identity':>9} {'location':>10} {'confidence':>12} {'timestamp':>11}")
print("-" * 76)
results = {
    "old value": v_old,
    "new value": v_new,
    "GDN2 b=1 (plain)": run_gdn2(1.0),
    "GDN2 b=0 (accum)": run_gdn2(0.0),
    "GDN2 masked error": run_masked(),
    "KDA (beta=1)": run_kda(),
}
for name, v in results.items():
    print(f"{name:>26} | {fmt(v)}")

print()
print("Untouched fields should survive as identity=1.00, confidence=0.90.")
for name, v in results.items():
    survived = abs(v[0] - 1.0) < 1e-3 and abs(v[2] - 0.9) < 1e-3
    print(f"  {name:>26}: untouched {'SURVIVE' if survived else 'destroyed'} ({v[0]:.2f}, {v[2]:.2f})")
