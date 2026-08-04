import torch.nn.functional as F
import torch


def sparse_delta_memory_recurrent(
    queries: torch.Tensor,
    query_indices: torch.Tensor,  # [T, R]
    keys: torch.Tensor,
    key_indices: torch.Tensor,  # [T, W]
    values: torch.Tensor,
    alphas: torch.Tensor,  # [T]
    betas: torch.Tensor,  # [T]
    memory_size: int,
    eps: float = 1e-8,
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
    memories = torch.zeros((T, memory_size, d_v)).to(device)
    memory = torch.zeros((memory_size, d_v)).to(device)
    keys = F.normalize(keys, 2, 1, eps)
    for i in range(T):
        decayed_state = memory.clone()
        old_values = decayed_state[key_indices[i]]
        new_values = alphas[i].unsqueeze(-1) * old_values
        selected_key = keys[i]
        prediction = selected_key @ new_values
        error = values[i] - prediction
        decayed_state[key_indices[i]] = new_values + betas[i] * keys[i].unsqueeze(
            -1
        ) * error.unsqueeze(0)
        memory = decayed_state

        outputs[i] = (
            queries[i].unsqueeze(-1)
            * memory[query_indices[i]]
        ).sum(dim=0)

        memories[i] = memory

    return outputs, memories


def dense_score_selection(
    write_scores: torch.Tensor,
    read_scores: torch.Tensor,
    W: int,
    R: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    write_weights, key_indices = write_scores.topk(W, dim=-1)
    read_weights, query_indices = read_scores.topk(R, dim=-1)

    return read_weights, query_indices, write_weights, key_indices

def product_key_topk_full(
    scores_a: torch.Tensor, 
    scores_b: torch.Tensor,
    k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_scores = scores_a.unsqueeze(-1) + scores_b.unsqueeze(-2) # outer sum
    flat_scores = pair_scores.flatten(start_dim=-2)
    return flat_scores.topk(k, dim=-1)

def product_key_topk_efficient(
    scores_a: torch.Tensor, 
    scores_b: torch.Tensor,  # [..., m]
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = scores_a.shape[-1]

    top_values_a, top_indices_a = scores_a.topk(k=k, dim=-1)
    top_values_b, top_indices_b = scores_b.topk(k=k, dim=-1)

    candidate_scores = (
        top_values_a.unsqueeze(-1)
        + top_values_b.unsqueeze(-2)
    )

    flat_candidate_scores = candidate_scores.flatten(start_dim=-2)

    top_scores, candidate_positions = flat_candidate_scores.topk(
        k=k,
        dim=-1,
    )

    positions_a = candidate_positions // k
    positions_b = candidate_positions % k

    # Converting candidate table positions back to original subkey indices.
    original_indices_a = torch.gather(
        top_indices_a,
        dim=-1,
        index=positions_a,
    )
    original_indices_b = torch.gather(
        top_indices_b,
        dim=-1,
        index=positions_b,
    )

    flattened_memory_indices = (
        original_indices_a * m
        + original_indices_b
    )

    return top_scores, flattened_memory_indices
    