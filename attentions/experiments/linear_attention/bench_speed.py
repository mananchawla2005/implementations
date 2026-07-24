import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.linear_attention import linear_attention_parallel, linear_attention_recurrent
from shared.plotting import plot_times, SEQ_LENS
from shared.benchmark_utils import time_it

d_k = 10
d_v = 10

recurrent_times = []
parallel_times = []

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()

    _, t_rec = time_it(linear_attention_recurrent, queries, keys, values)
    recurrent_times.append(t_rec)

    _, t_par = time_it(linear_attention_parallel, queries, keys, values)
    parallel_times.append(t_par)

    print(f"T={T:4d} | recurrent={t_rec:.4f}s | parallel={t_par:.4f}s")

plot_times(SEQ_LENS, {"Recurrent": recurrent_times, "Parallel": parallel_times}, title="Linear Attention Speed Benchmark")
