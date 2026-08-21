import numpy as np

def deterministic_moe_backward(contributions: list, num_ranks: int, num_tokens: int, dim: int) -> list:
    """
    Deterministically accumulate MoE backward-pass token gradients.

    Args:
        contributions: list of (rank, token_id, grad_list) tuples.
        num_ranks: number of source ranks.
        num_tokens: number of destination tokens.
        dim: gradient dimensionality.

    Returns:
        Nested list of shape (num_tokens, dim) with the accumulated gradient.
    """
    # Initialize per-rank buffers: shape (num_ranks, num_tokens, dim)
    rank_buffers = np.zeros((num_ranks, num_tokens, dim), dtype=np.float64)
    
    # Group contributions by rank
    rank_contributions = [[] for _ in range(num_ranks)]
    for rank, token_id, grad in contributions:
        rank_contributions[rank].append((token_id, grad))
    
    # For each rank, sort by token_id in ascending order (stable)
    # Stable sorting preserves original order for contributions with same token_id
    for rank in range(num_ranks):
        rank_contributions[rank].sort(key=lambda x: x[0])  # Sort by token_id
        
        # Accumulate into rank's buffer
        for token_id, grad in rank_contributions[rank]:
            rank_buffers[rank, token_id, :] += np.array(grad, dtype=np.float64)
    
    # Deterministic reduction: sum per-rank buffers in ascending rank order
    final_grad = np.zeros((num_tokens, dim), dtype=np.float64)
    for rank in range(num_ranks):
        final_grad += rank_buffers[rank]
    
    # Convert to nested list
    return final_grad.tolist()