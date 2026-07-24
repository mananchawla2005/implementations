import torch
import math

def causal_softmax_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    # shape is [T, d_k], [T, d_k], [T, d_v] -> [T, d_v]
    T, d_k = queries.shape
    _, d_v = values.shape
    output = torch.zeros((T, d_v))

    for i in range(T):
        attention_scores = queries[i] @ keys[: i + 1].T
        attention_scores = attention_scores / math.sqrt(d_k)
        max_row = torch.max(attention_scores)
        attention_scores = attention_scores - max_row
        sum_row = torch.sum(torch.exp(attention_scores))
        attention_scores = torch.exp(attention_scores) / sum_row
        output[i] = attention_scores @ values[: i + 1]
    return output
