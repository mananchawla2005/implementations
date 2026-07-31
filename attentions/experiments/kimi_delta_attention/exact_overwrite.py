import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent

d_k = 2
d_v = 2
T = 10

torch.manual_seed(0)
keys = torch.ones(T, d_k)
keys = F.normalize(keys, 2, 1)
values = torch.randn(T, d_v)
queries = keys.clone()
alphas = torch.randn(T, d_k)
betas = torch.ones(T)

_, out_states = kimi_delta_attention_recurrent(queries, keys, values, alphas, betas)

reads = torch.einsum('tkv,tk->tv', out_states, keys)
print(reads)
print(values)
print("\nMatch:", torch.allclose(values, reads))
