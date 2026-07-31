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
d_v = 3
T = 1

key = F.normalize(
    torch.tensor([1.0, 1.0]),
    p=2,
    dim=0,
)

queries = key.unsqueeze(0)  # [1, d_k]
keys = key.unsqueeze(0)     # [1, d_k]

values = torch.tensor([
    [10.0, 20.0, 30.0],
])

alphas = torch.ones(T, d_k)

# Disable erase so this test isolates the write gate.
erase_gates = torch.zeros(T, d_k)

write_gates = torch.tensor([
    [1.0, 0.0, 0.5],
])

expected_target = torch.tensor([10.0, 0.0, 15.0])

outputs, states = gated_delta_attention2_recurrent(
    queries=queries,
    keys=keys,
    values=values,
    alphas=alphas,
    erase_gates=erase_gates,
    write_gates=write_gates,
)

written_target = write_gates[0] * values[0]
retrieved_value = states[0].T @ key

# Since S_0 = k z^T and ||k|| = 1:
#
# S_0^T k = z (k^T k) = z.
expected_state = torch.outer(key, expected_target)

print("Value:")
print(values[0])

print("\nWrite gate:")
print(write_gates[0])

print("\nWritten target z = w . v:")
print(written_target)

print("\nExpected target:")
print(expected_target)

print("\nResulting state:")
print(states[0])

print("\nExpected state k z^T:")
print(expected_state)

print("\nRetrieved value:")
print(retrieved_value)

print(
    "\nTarget correct:",
    torch.allclose(
        written_target,
        expected_target,
        atol=1e-6,
    ),
)

print(
    "State correct:",
    torch.allclose(
        states[0],
        expected_state,
        atol=1e-6,
    ),
)

print(
    "Retrieval correct:",
    torch.allclose(
        retrieved_value,
        expected_target,
        atol=1e-6,
    ),
)

