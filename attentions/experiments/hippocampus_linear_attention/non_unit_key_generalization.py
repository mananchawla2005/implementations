import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 16
d_v = 8
n_seeds = 200

print(f"{'norm(k)':>8} | {'true m_t':>10} | {'beta*||e||':>11} | {'beta*||k||*||e||':>17} | {'rel err(simpl.)':>15}")
print("-" * 75)

for norm_target in [0.3, 0.7, 1.0, 1.4, 2.5]:
    worst_simple = 0.0
    worst_full = 0.0
    for seed in range(n_seeds):
        g = torch.Generator().manual_seed(seed)

        S_prev = torch.randn(d_k, d_v, generator=g)
        k = torch.randn(d_k, generator=g)
        k = k / k.norm() * norm_target            # control ||k_t||
        v = torch.randn(d_v, generator=g)
        alpha = torch.rand(1, generator=g).item() * 0.9 + 0.1
        beta = torch.rand(1, generator=g).item() * 0.9 + 0.1

        e = v - S_prev.T @ k                       
        S_next = alpha * S_prev + beta * torch.outer(k, e)

        true_m = (S_next - alpha * S_prev).norm("fro")
        simple = beta * e.norm()                   
        full = beta * k.norm() * e.norm()       

        worst_simple = max(worst_simple, abs(true_m - simple) / true_m)
        worst_full = max(worst_full, abs(true_m - full) / true_m)

    print(
        f"{norm_target:>8.2f} | {true_m:>10.4f} | {simple:>11.4f} | "
        f"{full:>17.4f} | {worst_simple:>14.2%} (full: {worst_full:.2e})"
    )

print()
print("Simplification beta*||e|| is exact only when ||k_t|| = 1.")
exact_at_unit = True
broken_at_other = False
for seed in range(n_seeds):
    g = torch.Generator().manual_seed(seed)
    S_prev = torch.randn(d_k, d_v, generator=g)
    v = torch.randn(d_v, generator=g)
    beta = torch.rand(1, generator=g).item() * 0.9 + 0.1

    k_unit = F.normalize(torch.randn(d_k, generator=g), dim=-1)
    e_unit = v - S_prev.T @ k_unit
    S_unit = S_prev + beta * torch.outer(k_unit, e_unit)
    if abs((S_unit - S_prev).norm("fro") - beta * e_unit.norm()) > 1e-4:
        exact_at_unit = False

    k_other = torch.randn(d_k, generator=g) * 2.0
    e_other = v - S_prev.T @ k_other
    S_other = S_prev + beta * torch.outer(k_other, e_other)
    true_m = (S_other - S_prev).norm("fro")
    if abs(true_m - beta * e_other.norm()) / true_m > 0.01:
        broken_at_other = True

print("Holds for all seeds at ||k||=1:", exact_at_unit)
print("Fails for at least one non-unit key seed:", broken_at_other)

assert exact_at_unit, "simplification should be exact at unit norm"
assert broken_at_other, "simplification must fail for some non-unit key"
