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

keys = torch.stack([
    key,  # store
    key,  # delete
])

queries = keys.clone()

# The value on the deletion step is irrelevant because write gate is zero.
values = torch.stack([
    stored_value,
    torch.tensor([123.0, 456.0]),
])

alphas = torch.ones(T, d_k)

erase_gates = torch.stack([
    torch.zeros(d_k),  # store without erasing
    torch.ones(d_k),   # delete the association
])

write_gates = torch.stack([
    torch.ones(d_v),   # write stored_value
    torch.zeros(d_v),  # write nothing
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
read_after_delete = states[1].T @ key

print("Stored value:      ", stored_value)
print("After store:       ", read_after_store)
print("After delete:      ", read_after_delete)

print(
    "\nStore matches:",
    torch.allclose(
        read_after_store,
        stored_value,
        atol=1e-6,
    ),
)

print(
    "Deletion is zero:",
    torch.allclose(
        read_after_delete,
        torch.zeros_like(stored_value),
        atol=1e-6,
    ),
)