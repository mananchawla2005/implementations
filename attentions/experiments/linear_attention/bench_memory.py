import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.linear_attention import linear_attention_recurrent, linear_attention_parallel
from shared.plotting import SEQ_LENS

d_k = 128
d_v = 128

print(f"{'T':>4} | {'recurrent (MB)':>14} | {'parallel (MB)':>14}")
print("-" * 40)

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        keys_cuda = keys.cuda()
        values_cuda = values.cuda()
        queries_cuda = queries.cuda()

        linear_attention_recurrent(queries_cuda, keys_cuda, values_cuda)
        mem_rec = torch.cuda.max_memory_allocated() / 1024 ** 2

        torch.cuda.reset_peak_memory_stats()
        linear_attention_parallel(queries_cuda, keys_cuda, values_cuda)
        mem_par = torch.cuda.max_memory_allocated() / 1024 ** 2

        print(f"{T:4d} | {mem_rec:>13.2f} MB | {mem_par:>13.2f} MB")
    else:
        print(f"{T:4d} | {'N/A (no CUDA)':>14} | {'N/A (no CUDA)':>14}")
