import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.softmax_attention import causal_softmax_attention
from shared.plotting import SEQ_LENS

d_k = 128
d_v = 128

print(f"{'T':>4} | {'peak memory (MB)':>16}")
print("-" * 26)

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        causal_softmax_attention(queries.cuda(), keys.cuda(), values.cuda())
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"{T:4d} | {mem:>14.2f} MB")
    else:
        print(f"{T:4d} | {'N/A (no CUDA)':>16}")
