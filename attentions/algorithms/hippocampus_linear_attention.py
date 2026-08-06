import torch.nn.functional as F
import math
import torch

def update_topk_cache_functional(
    persistent_keys: torch.Tensor,  # [M, d_k]
    persistent_values: torch.Tensor,  # [M, d_v]
    persistent_scores: torch.Tensor,  # [M]
    persistent_positions: torch.Tensor,  # [M]
    new_keys: torch.Tensor,  # [C, d_k]
    new_values: torch.Tensor,  # [C, d_v]
    new_scores: torch.Tensor,  # [C]
    new_positions: torch.Tensor,  # [C]
    capacity: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    candidate_keys = torch.cat(
        [persistent_keys, new_keys],
        dim=0,
    )
    candidate_values = torch.cat(
        [persistent_values, new_values],
        dim=0,
    )
    candidate_scores = torch.cat(
        [persistent_scores, new_scores],
        dim=0,
    )
    candidate_positions = torch.cat(
        [persistent_positions, new_positions],
        dim=0,
    )

    keep_count = min(capacity, candidate_scores.shape[0])

    if keep_count == 0:
        return (
            candidate_keys[:0],
            candidate_values[:0],
            candidate_scores[:0],
            candidate_positions[:0],
        )

    keep_indices = torch.topk(
        candidate_scores,
        k=keep_count,
        dim=0,
        largest=True,
        sorted=False,
    ).indices

    return (
        candidate_keys.index_select(0, keep_indices),
        candidate_values.index_select(0, keep_indices),
        candidate_scores.index_select(0, keep_indices),
        candidate_positions.index_select(0, keep_indices),
    )


def cache_attention_chunk(
    chunk_queries: torch.Tensor,       # [C, d_k]
    persistent_keys: torch.Tensor,     # [M, d_k]
    persistent_values: torch.Tensor,   # [M, d_v]
    chunk_keys: torch.Tensor,          # [C, d_k]
    chunk_values: torch.Tensor,        # [C, d_v]
    sink_logit: torch.Tensor # scaler per head
) -> torch.Tensor:
    """
    Returns:
        outputs: [C, d_v]
    """
    C, d_k = chunk_queries.shape
    M = persistent_keys.shape[0]

    visible_keys = torch.cat(
        [
            persistent_keys,
            chunk_keys
        ],
        dim=0,
    )  # [M + C , d_k]

    visible_values = torch.cat(
        [
            persistent_values,
            chunk_values
        ],
        dim=0,
    )  # [M + C, d_v]

    logits = (
        chunk_queries @ visible_keys.transpose(0, 1)
        / math.sqrt(d_k)
    )  # [C, M + C]

    persistent_mask = torch.ones(
        C,
        M,
        dtype=torch.bool,
        device=chunk_queries.device,
    )

    block_mask = torch.tril(
        torch.ones(
            C,
            C,
            dtype=torch.bool,
            device=chunk_queries.device,
        )
    )

    visible_mask = torch.cat(
        [
            persistent_mask,
            block_mask,
        ],
        dim=-1,
    )  # [C, M + C]

    logits = logits.masked_fill(
        ~visible_mask,
        torch.finfo(logits.dtype).min,
    )
    null_logits = sink_logit.to(dtype=logits.dtype, device=logits.device).expand(C, 1)
    all_logits = torch.cat([logits, null_logits], dim=-1)
    weights = torch.softmax(all_logits, dim=-1)

    cache_weights = weights[:, :-1] # equivalent to adding a zero value into visible value and multiplying

    return cache_weights @ visible_values


def hola_attention_chunked(
    queries: torch.Tensor,  # [T, d_k]
    keys: torch.Tensor,  # [T, d_k]
    cache_queries: torch.Tensor,  # [T, d_k], cache-scaled
    cache_keys: torch.Tensor,  # [T, d_k], cache-scaled
    sink_logit: torch.Tensor,
    values: torch.Tensor,  # [T, d_v]
    betas: torch.Tensor,  # [T]
    alphas: torch.Tensor,  # [T, d_k]
    cache_gates: torch.Tensor,  # [T]
    capacity: int,
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

    n = T // chunk_size

    q_chunks = queries.reshape(n, chunk_size, d_k)
    k_chunks = keys.reshape(n, chunk_size, d_k)
    v_chunks = values.reshape(n, chunk_size, d_v)

    beta_chunks = betas.reshape(n, chunk_size)
    alpha_chunks = alphas.reshape(n, chunk_size, d_k)
    checkpoints = torch.zeros(
        n + 1,
        d_k,
        d_v,
        device=device,
        dtype=values.dtype,
    )
    checkpoints[0].zero_()
    # g_r = alpha_0 * alpha_1 * ... * alpha_r
    # independently inside every chunk.
    gamma_prefix = torch.cumprod(
        alpha_chunks,
        dim=1,
    )  # [C, d_k]
    write_keys = k_chunks / gamma_prefix.clamp_min(eps)  # A: [C, d_k]
    read_keys = k_chunks * gamma_prefix  # B: [C, d_k]
    scaled_queries = gamma_prefix * q_chunks  # Qbar: [C, d_k]

    interaction = read_keys @ write_keys.transpose(-1, -2)  # [C, C]

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
        beta_chunks.unsqueeze(-1) * v_chunks,
        upper=False,
        unitriangular=True,
    )

    W = torch.linalg.solve_triangular(
        L,
        beta_chunks.unsqueeze(-1) * read_keys,
        upper=False,
        unitriangular=True,
    )

    corrected_values_per_chunk = torch.empty(
        n,
        chunk_size,
        d_v,
        device=device,
        dtype=values.dtype,
    )
    persistent_keys = torch.empty(
        0,
        d_k,
        device=device,
        dtype=cache_keys.dtype,
    )

    persistent_values = torch.empty(
        0,
        d_v,
        device=device,
        dtype=values.dtype,
    )

    persistent_scores = torch.empty(
        0,
        device=device
    )

    persistent_positions = torch.empty(
        0,
        device=device,
        dtype=torch.long,
    )
    cache_outputs = torch.zeros(
        n,
        chunk_size,
        d_v,
        device=device,
        dtype=values.dtype,
    )
    for i in range(n):

        R_in = checkpoints[i]

        corrected_values = U[i] - W[i] @ R_in

        normalized_R_out = R_in + write_keys[i].transpose(-1, -2) @ corrected_values

        chunk_gate = gamma_prefix[i, -1]

        S_out = chunk_gate.unsqueeze(-1) * normalized_R_out

        corrected_values_per_chunk[i] = corrected_values
        checkpoints[i + 1] = S_out

        chunk_surprise = corrected_values.norm(dim=-1)  # [C]
        chunk_start = i * chunk_size
        chunk_end = chunk_start + chunk_size
        chunk_cache_keys = cache_keys[chunk_start:chunk_end]
        chunk_cache_queries = cache_queries[chunk_start:chunk_end]
        chunk_cache_values = values[chunk_start:chunk_end]

        cache_outputs[i] = cache_attention_chunk(
            chunk_queries=chunk_cache_queries,
            persistent_keys=persistent_keys,
            persistent_values=persistent_values,
            chunk_keys=chunk_cache_keys,
            chunk_values=chunk_cache_values,
            sink_logit=sink_logit
        )
        chunk_positions = torch.arange(
            chunk_start,
            chunk_end,
            device=device,
            dtype=torch.long,
        )
        (
            persistent_keys,
            persistent_values,
            persistent_scores,
            persistent_positions,
        ) = update_topk_cache_functional(
            persistent_keys=persistent_keys,
            persistent_values=persistent_values,
            persistent_scores=persistent_scores,
            persistent_positions=persistent_positions,
            new_keys=chunk_cache_keys,
            new_values=chunk_cache_values,
            new_scores=chunk_surprise,
            new_positions=chunk_positions,
            capacity=capacity,
        )

    checkpoint_outputs = scaled_queries @ checkpoints[:-1]  # [n, C, d_v]

    local_scores = torch.tril(
        scaled_queries @ write_keys.transpose(-1, -2),
        diagonal=0,
    )  # [n, C, C]

    local_outputs = local_scores @ corrected_values_per_chunk

    # State at position r is multiplied by
    # alpha_0 * ... * alpha_r.
    outputs = (
        checkpoint_outputs
        + local_outputs
        + cache_gates.reshape(n, chunk_size, 1) * cache_outputs
    )

    return outputs.reshape(T, d_v)
