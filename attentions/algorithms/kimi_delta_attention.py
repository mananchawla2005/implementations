import torch.nn.functional as F
import torch

def kimi_delta_attention_recurrent(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    eps: float = 1e-8,
    initial_state: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
        outputs: [T, d_v]
        states:  [T, d_k, d_v]
    """
    T, d_k = queries.shape
    _, d_v = values.shape
    device = queries.device
    outputs = torch.zeros((T, d_v)).to(device)
    states = torch.zeros((T, d_k, d_v)).to(device)
    if initial_state is None:
        state = torch.zeros((d_k, d_v)).to(device)
    else:
        state = initial_state.clone()
    keys = F.normalize(keys, 2, 1, eps)
    for i in range(T):
        decayed_state = alphas[i].unsqueeze(-1) * state
        prediction = decayed_state.T @ keys[i]
        error = values[i] - prediction
        state = decayed_state + betas[i] * torch.outer(keys[i], error)
        outputs[i] = state.T @ queries[i]
        states[i] = state

    return outputs, states

def kimi_delta_attention_chunked(
    queries: torch.Tensor, # [T, d_k]
    keys: torch.Tensor, # [T, d_k]
    values: torch.Tensor, # [T, d_v]
    betas: torch.Tensor, # [T]
    alphas: torch.Tensor, # [T, d_k]
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
    alpha_chunks = alphas.reshape(n, chunk_size, d_k)
    checkpoints = torch.empty((n + 1, d_k, d_v)).to(device)
    checkpoints[0].zero_()
    # g_r = alpha_0 * alpha_1 * ... * alpha_r
    # independently inside every chunk.
    gamma_prefix = torch.cumprod(
        alpha_chunks,
        dim=1,
    ) # [C, d_k]
    write_keys = k_chunks / gamma_prefix.clamp_min(eps) # A: [C, d_k]
    read_keys = k_chunks * gamma_prefix # B: [C, d_k]
    scaled_queries = gamma_prefix * q_chunks # Qbar: [C, d_k]

    interaction = read_keys @ write_keys.transpose(-1, -2) # [C, C]

    coeff = beta_chunks.unsqueeze(-1) * interaction

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

    U = torch.linalg.solve_triangular(
        L,
        beta_chunks.unsqueeze(-1)*v_chunks,
        upper=False,
        unitriangular=True,
    )

    W = torch.linalg.solve_triangular(
        L,
        beta_chunks.unsqueeze(-1)*read_keys,
        upper=False,
        unitriangular=True,
    )

    corrected_values_per_chunk = torch.empty((n, chunk_size, d_v)).to(device)

    for i in range(n):
        R_in = checkpoints[i]

        corrected_values = U[i] - W[i] @ R_in

        normalized_R_out = R_in + write_keys[i].transpose(-1, -2) @ corrected_values

        chunk_gate = gamma_prefix[i, -1]

        S_out = chunk_gate.unsqueeze(-1) * normalized_R_out

        corrected_values_per_chunk[i] = corrected_values
        checkpoints[i + 1] = S_out

    checkpoint_outputs = scaled_queries @ checkpoints[:-1]  # [n, C, d_v]

    local_scores = torch.tril(
        scaled_queries @ write_keys.transpose(-1, -2),
        diagonal=0,
    )  # [n, C, C]

    local_outputs = local_scores @ corrected_values_per_chunk

    # State at position r is multiplied by
    # alpha_0 * ... * alpha_r.
    outputs = (checkpoint_outputs + local_outputs)

    return outputs.reshape(T, d_v)
