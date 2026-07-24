import torch.nn.functional as F
import torch

def combine(
    A_left: torch.Tensor,
    B_left: torch.Tensor,
    A_right: torch.Tensor,
    B_right: torch.Tensor,
):
    A_combined = A_left @ A_right
    B_combined = B_left @ A_right + B_right
    return A_combined, B_combined

def prefix_scan(
    A: torch.Tensor,
    B: torch.Tensor,
):
    """
    A: [T, d_k, d_k]
    B: [T, d_v, d_k]

    Returns:
        A_prefix: [T, d_k, d_k]
        B_prefix: [T, d_v, d_k]
    """
    T, _, _ = A.shape

    A_prefix = A.clone()
    B_prefix = B.clone()

    distance = 1

    while distance < T:
        A_old = A_prefix.clone()
        B_old = B_prefix.clone()

        A_left = A_old[:-distance]
        B_left = B_old[:-distance]

        A_right = A_old[distance:]
        B_right = B_old[distance:]

        A_combined, B_combined = combine(
            A_left,
            B_left,
            A_right,
            B_right,
        )

        A_prefix[distance:] = A_combined
        B_prefix[distance:] = B_combined

        distance *= 2

    return A_prefix, B_prefix

def delta_attention_recurrent(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, betas: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Return:
        outputs: [T, d_v]
        states:  [T, d_v, d_k]
    """
    # shape is [T, d_k], [T, d_k], [T, d_v] -> [T, d_v]
    T, d_k = queries.shape
    _, d_v = values.shape
    device = queries.device
    outputs = torch.zeros((T, d_v)).to(device)
    states = torch.zeros((T, d_v, d_k)).to(device)
    keys = F.normalize(keys, 2, 1, eps)
    for i in range(T):
        if i == 0:
            states[i] = betas[i]*torch.outer(values[i], keys[i])
        else:
            states[i] = states[i - 1] + betas[i]*torch.outer(values[i]-(states[i-1]@keys[i]), keys[i])
        outputs[i] = states[i] @ queries[i]

    return outputs, states

def delta_attention_parallel(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, betas: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Return:
        outputs: [T, d_v]
        states:  [T, d_v, d_k]
    """
    # shape is [T, d_k], [T, d_k], [T, d_v] -> [T, d_v]
    T, d_k = queries.shape
    _, d_v = values.shape
    device = queries.device
    outputs = torch.zeros((T, d_v)).to(device)
    states = torch.zeros((T, d_v, d_k)).to(device)
    keys = F.normalize(keys, 2, 1, eps)

    writes = torch.einsum(
        "tv,tk->tvk", values, keys
    )  # each write[t] is exactly outer(values[t], keys[t])
    key_outer = torch.einsum("ti,tj->tij", keys, keys)
    identity = torch.eye(d_k)
    A = identity - betas[:, None, None] * key_outer
    assert(A.shape == (T, d_k, d_k))
    B = betas[:, None, None] * writes
    assert(B.shape == (T, d_v, d_k))
    _, states = prefix_scan(A, B)


    outputs = torch.einsum("tvk,tk->tv", states, queries)

    return outputs, states

def delta_attention_parallel_nlms(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, betas: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Return:
        outputs: [T, d_v]
        states:  [T, d_v, d_k]
    """
    # shape is [T, d_k], [T, d_k], [T, d_v] -> [T, d_v]
    T, d_k = queries.shape
    _, d_v = values.shape
    device = queries.device
    outputs = torch.zeros((T, d_v)).to(device)
    states = torch.zeros((T, d_v, d_k)).to(device)
    keys_used = keys
    key_norm_sq = keys.square().sum(dim=-1)
    gamma = betas / (key_norm_sq + eps)
    writes = torch.einsum(
        "tv,tk->tvk", values, keys_used
    )  # each write[t] is exactly outer(values[t], keys[t])
    key_outer = torch.einsum("ti,tj->tij", keys_used, keys_used)
    identity = torch.eye(d_k)
    A = identity - gamma[:, None, None] * key_outer
    assert(A.shape == (T, d_k, d_k))
    B = gamma[:, None, None] * writes
    assert(B.shape == (T, d_v, d_k))
    _, states = prefix_scan(A, B)

    outputs = torch.einsum("tvk,tk->tv", states, queries)

    return outputs, states