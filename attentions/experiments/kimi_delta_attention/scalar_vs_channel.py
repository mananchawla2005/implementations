import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.inference import GatedDeltaNet, KimiDeltaAttention

T = 110

k_short = torch.tensor([1.0, 0.0])
k_long = torch.tensor([0.0, 1.0])
v_short = torch.tensor([2.0, 0.0])
v_long = torch.tensor([0.0, 2.0])

print(f"{'Model':>18} | {'short@5':>8} | {'long@100':>9}")
print("-" * 41)

for scalar_alpha in [0.8, 0.99]:
    m = GatedDeltaNet(2, 2)
    stores = []
    m.step(k_short, k_short, v_short, alpha=1.0, beta=1.0); stores.append(m.state.clone())
    m.step(k_long, k_long, v_long, alpha=1.0, beta=1.0); stores.append(m.state.clone())
    m.step(k_short, k_short, torch.tensor([-3.0, 1.0]), alpha=scalar_alpha, beta=1.0); stores.append(m.state.clone())
    for t in range(3, T):
        m.step(k_short, k_short, torch.zeros(2), alpha=scalar_alpha, beta=0.0)
        stores.append(m.state.clone())
    r5 = (stores[5] @ k_short).norm().item()
    r100 = (stores[100] @ k_long).norm().item()
    print(f"{'scalar '+str(scalar_alpha):>18} | {r5:>7.3f}  | {r100:>8.3f}")

alpha_vec = torch.tensor([0.8, 0.99])
m = KimiDeltaAttention(2, 2)
stores = []
m.step(k_short, k_short, v_short, alpha=torch.ones(2), beta=1.0); stores.append(m.state.clone())
m.step(k_long, k_long, v_long, alpha=torch.ones(2), beta=1.0); stores.append(m.state.clone())
m.step(k_short, k_short, torch.tensor([-3.0, 1.0]), alpha=alpha_vec, beta=1.0); stores.append(m.state.clone())
for t in range(3, T):
    m.step(k_short, k_short, torch.zeros(2), alpha=alpha_vec, beta=0.0)
    stores.append(m.state.clone())
r5 = (stores[5].T @ k_short).norm().item()
r100 = (stores[100].T @ k_long).norm().item()
print(f"{'KDA 0.8/0.99':>18} | {r5:>7.3f}  | {r100:>8.3f}")
