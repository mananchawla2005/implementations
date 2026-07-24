import torch

def gated_linear_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, alphas: torch.Tensor
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
    P = alphas.cumprod(dim=0)
    P = P.add(1e-10)
    writes = torch.einsum(
        "tv,tk->tvk", values, keys
    )  # each write[t] is exactly outer(values[t], keys[t])
    writes = writes / P[:, None, None]
    normalised_states = writes.cumsum(dim=0)
    states = normalised_states * P[:, None, None]
    outputs = torch.einsum("tvk,tk->tv", states, queries)

    return outputs, states
