import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.softmax_attention import causal_softmax_attention
from shared.plotting import plot_times, SEQ_LENS
from shared.benchmark_utils import time_it

d_k = 10
d_v = 10

times = []

for T in SEQ_LENS:
    keys = torch.randn(T, d_k)
    values = torch.randn(T, d_v)
    queries = keys.clone()

    _, elapsed = time_it(causal_softmax_attention, queries, keys, values)
    times.append(elapsed)

    print(f"{elapsed:.4f}s for seq_len={T}")

plot_times(SEQ_LENS, {"Softmax": times}, title="Softmax Attention Speed Benchmark")
