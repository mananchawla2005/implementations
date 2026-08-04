import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.sparse_delta_memory import (
    product_key_topk_full,
    product_key_topk_efficient,
)


N_values = [16, 64, 256, 1024]
k_values = [1, 2, 4, 8, 16, 32]
seeds = 20

print(f"{'N':>5} {'k':>3} | mismatches")
all_ok = True
for N in N_values:
    m = int(N ** 0.5) 
    for k in k_values:
        if k > m:
            continue
        for seed in range(seeds):
            g = torch.Generator().manual_seed(seed + N * 100 + k)
            s1 = torch.randn(m, generator=g)
            s2 = torch.randn(m, generator=g)
            full_vals, full_idx = product_key_topk_full(s1, s2, k)
            eff_vals, eff_idx = product_key_topk_efficient(s1, s2, k)
            if not torch.equal(full_idx, eff_idx):
                print(f"{N:>5} {k:>3} | seed {seed}: index mismatch")
                all_ok = False
            if not torch.allclose(full_vals, eff_vals, atol=1e-6):
                print(f"{N:>5} {k:>3} | seed {seed}: value mismatch")
                all_ok = False
print(f"\nAll seeds matched: {all_ok}")

assert all_ok, "product-key top-k mismatch"
