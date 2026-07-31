import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from algorithms.gated_delta_attention2 import (
    gated_delta_attention2_recurrent,
)

d_k = 2
d_v = 2
T = 2

key = F.normalize(
    torch.tensor([1.0, 1.0]),
    p=2,
    dim=0,
)

stored_value = torch.tensor([3.0, -2.0])
stored_value2 = torch.tensor([123.0, 456.0])
keys = torch.stack([
    key,  # store
    key,  # store
])

queries = keys.clone()

values = torch.stack([
    stored_value,
    stored_value2,
])

alphas = torch.ones(T, d_k)

erase_gates = torch.stack([
    torch.zeros(d_k),  
    torch.zeros(d_k),   
])

write_gates = torch.stack([
    torch.ones(d_v),  
    torch.ones(d_v),  
])

outputs, states = gated_delta_attention2_recurrent(
    queries,
    keys,
    values,
    alphas,
    erase_gates,
    write_gates,
)

read_after_store = states[0].T @ key
read_after_store2 = states[1].T @ key

print("Stored value:      ", stored_value)
print("After store:       ", read_after_store)
print("After store2:      ", read_after_store2)

print(
    "\nStore1 matches:",
    torch.allclose(
        read_after_store,
        stored_value,
        atol=1e-6,
    ),
)

print(
    "Store2 matches:",
    torch.allclose(
        read_after_store2,
        stored_value+stored_value2,
        atol=1e-6,
    ),
)
