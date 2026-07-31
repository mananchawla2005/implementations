import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from algorithms.gated_delta_attention2 import (
    gated_delta_attention2_recurrent,
)


def run_partial_erase(
    initial_state: torch.Tensor,
    key: torch.Tensor,
    erase_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_k, d_v = initial_state.shape

    queries = key.unsqueeze(0)
    keys = key.unsqueeze(0)

    values = torch.zeros(1, d_v)

    alphas = torch.ones(1, d_k)
    erase_gates = erase_gate.unsqueeze(0)
    write_gates = torch.zeros(1, d_v)

    _, states = gated_delta_attention2_recurrent(
        queries=queries,
        keys=keys,
        values=values,
        alphas=alphas,
        erase_gates=erase_gates,
        write_gates=write_gates,
        initial_state=initial_state,
    )

    erase_key = erase_gate * key
    erased_value = initial_state.T @ erase_key
    final_state = states[0]

    return erase_key, erased_value, final_state


sqrt_two = torch.sqrt(torch.tensor(2.0))

key = torch.tensor([1.0, 1.0]) / sqrt_two

b_left = torch.tensor([1.0, 0.0])
b_right = torch.tensor([0.0, 1.0])


initial_state = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0],
])

(
    e_left,
    erased_value_left,
    state_after_left,
) = run_partial_erase(
    initial_state=initial_state,
    key=key,
    erase_gate=b_left,
)

(
    e_right,
    erased_value_right,
    state_after_right,
) = run_partial_erase(
    initial_state=initial_state,
    key=key,
    erase_gate=b_right,
)

expected_e_left = torch.tensor([
    1.0 / sqrt_two,
    0.0,
])

expected_e_right = torch.tensor([
    0.0,
    1.0 / sqrt_two,
])

expected_erased_left = initial_state[0] / sqrt_two

expected_erased_right = initial_state[1] / sqrt_two

expected_state_left = (
    initial_state
    - torch.outer(key, expected_erased_left)
)

expected_state_right = (
    initial_state
    - torch.outer(key, expected_erased_right)
)

initial_read = initial_state.T @ key
read_after_left = state_after_left.T @ key
read_after_right = state_after_right.T @ key

print("Key:")
print(key)

print("\nInitial state:")
print(initial_state)

print("\nInitial retrieval S^T k:")
print(initial_read)

print("\n--- Gate b = [1, 0] ---")

print("Erase key e = b . k:")
print(e_left)

print("Value read for deletion S^T e:")
print(erased_value_left)

print("State after deletion:")
print(state_after_left)

print("Retrieval after deletion:")
print(read_after_left)

print("\n--- Gate b = [0, 1] ---")

print("Erase key e = b . k:")
print(e_right)

print("Value read for deletion S^T e:")
print(erased_value_right)

print("State after deletion:")
print(state_after_right)

print("Retrieval after deletion:")
print(read_after_right)

print("\nChecks:")

print(
    "Left erase key correct:",
    torch.allclose(
        e_left,
        expected_e_left,
        atol=1e-6,
    ),
)

print(
    "Right erase key correct:",
    torch.allclose(
        e_right,
        expected_e_right,
        atol=1e-6,
    ),
)

print(
    "Left gate reads first state row:",
    torch.allclose(
        erased_value_left,
        expected_erased_left,
        atol=1e-6,
    ),
)

print(
    "Right gate reads second state row:",
    torch.allclose(
        erased_value_right,
        expected_erased_right,
        atol=1e-6,
    ),
)

print(
    "Left final state correct:",
    torch.allclose(
        state_after_left,
        expected_state_left,
        atol=1e-6,
    ),
)

print(
    "Right final state correct:",
    torch.allclose(
        state_after_right,
        expected_state_right,
        atol=1e-6,
    ),
)

print(
    "Different portions removed:",
    not torch.allclose(
        state_after_left,
        state_after_right,
        atol=1e-6,
    ),
)
