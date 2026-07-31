import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.kimi_delta_attention import kimi_delta_attention_recurrent

d_k = 2
d_v = 2
T = 1

torch.manual_seed(0)
keys = torch.randn(T, d_k)
values = torch.randn(T, d_v)
queries = keys.clone()
alphas = torch.tensor([[0, 1]])
betas = torch.zeros(T)

initial_state = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
_, out_states = kimi_delta_attention_recurrent(queries, keys, values, alphas, betas, initial_state=initial_state)

print("\nFunction result:")
print(out_states[0])
print("\nMatch:", torch.allclose(torch.tensor([[0., 0.], [0., 20.]]), out_states[0]))
