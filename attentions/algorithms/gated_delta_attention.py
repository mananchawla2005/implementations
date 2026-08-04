import torch.nn.functional as F
import torch


def combine(
    A_left: torch.Tensor,
    B_left: torch.Tensor,
    A_right: torch.Tensor,
    B_right: torch.Tensor,
):
    A_combined = A_left @ A_right
    B_combined = (B_left @ A_right) + B_right
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


def gated_delta_attention_parallel(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    A = alphas[:, None, None] * (identity - betas[:, None, None] * key_outer)
    assert A.shape == (T, d_k, d_k)
    B = betas[:, None, None] * writes
    assert B.shape == (T, d_v, d_k)

    _, states = prefix_scan(A, B)

    outputs = torch.einsum("tvk,tk->tv", states, queries)

    return outputs, states


def gated_delta_attention_chunked(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    chunk_size: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns:
        outputs: [T, d_v]
    """
    T, d_k = queries.shape
    _, d_v = values.shape
    device = queries.device

    keys = F.normalize(
        keys,
        p=2,
        dim=-1,
        eps=eps,
    )

    n = T // chunk_size

    q_chunks = queries.reshape(n, chunk_size, d_k)
    k_chunks = keys.reshape(n, chunk_size, d_k)
    v_chunks = values.reshape(n, chunk_size, d_v)

    beta_chunks = betas.reshape(n, chunk_size)
    alpha_chunks = alphas.reshape(n, chunk_size)
    checkpoints = torch.empty((n + 1, d_v, d_k)).to(device)
    checkpoints[0].zero_()
    # g_r = alpha_0 * alpha_1 * ... * alpha_r
    # independently inside every chunk.
    alpha_prefix = torch.cumprod(
        alpha_chunks,
        dim=-1,
    )

    gram = k_chunks @ k_chunks.transpose(-1, -2)

    coeff = beta_chunks.unsqueeze(-1) * gram

    lower = torch.tril(
        coeff,
        diagonal=-1,
    )

    L = (
        torch.eye(
            chunk_size,
            device=device,
        ).unsqueeze(0)
        + lower
    )

    beta_k = beta_chunks.unsqueeze(-1) * k_chunks

    # beta_t v_t  ->  (beta_t / g_t) v_t
    scaled_beta_v = (beta_chunks / alpha_prefix.clamp_min(eps)).unsqueeze(-1) * v_chunks

    W = torch.linalg.solve_triangular(
        L,
        beta_k,
        upper=False,
        unitriangular=True,
    )

    U = torch.linalg.solve_triangular(
        L,
        scaled_beta_v,
        upper=False,
        unitriangular=True,
    )

    corrected_values_per_chunk = torch.empty((n, chunk_size, d_v)).to(device)

    for i in range(n):
        S_in = checkpoints[i]

        corrected_values = U[i] - W[i] @ S_in.transpose(-1, -2)

        normalized_S_out = S_in + corrected_values.transpose(-1, -2) @ k_chunks[i]

        chunk_gate = alpha_prefix[i, -1]

        S_out = chunk_gate * normalized_S_out

        corrected_values_per_chunk[i] = corrected_values
        checkpoints[i + 1] = S_out

    checkpoint_outputs = q_chunks @ checkpoints[:-1].transpose(-1, -2)  # [n, C, d_v]

    local_scores = torch.tril(
        q_chunks @ k_chunks.transpose(-1, -2),
        diagonal=0,
    )  # [n, C, C]

    local_outputs = local_scores @ corrected_values_per_chunk

    # State at position r is multiplied by
    # alpha_0 * ... * alpha_r.
    outputs = alpha_prefix.unsqueeze(-1) * (checkpoint_outputs + local_outputs)

    return outputs.reshape(T, d_v)
