import torch

def linear_attention_recurrent(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
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

    for i in range(T):
        if i == 0:
            states[i] = torch.outer(values[i], keys[i])
        else:
            states[i] = states[i - 1] + torch.outer(values[i], keys[i])
        outputs[i] = states[i] @ queries[i]

    return outputs, states


def linear_attention_parallel(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
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

    writes = torch.einsum(
        "tv,tk->tvk", values, keys
    )  # each write[t] is exactly outer(values[t], keys[t])
    states = writes.cumsum(dim=0)
    outputs = torch.einsum("tvk,tk->tv", states, queries)

    return outputs, states
