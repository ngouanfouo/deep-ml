import torch

def prune_attention_heads_pytorch(
    attention_weights: torch.Tensor,
    head_importance_scores: torch.Tensor,
    pruning_ratio: float
) -> tuple[torch.Tensor, list[int]]:
    """
    Prune attention heads using PyTorch.
    
    Args:
        attention_weights: Shape (num_heads, seq_len, seq_len)
        head_importance_scores: Shape (num_heads,)
        pruning_ratio: Fraction to prune (0.0 to 1.0)
    
    Returns:
        Tuple of (pruned_attention_weights, kept_head_indices)
    """
    num_heads = attention_weights.shape[0]
    
    # Clamp pruning_ratio between 0 and 1
    pruning_ratio = max(0.0, min(1.0, pruning_ratio))
    
    # Calculate number of heads to keep
    num_heads_to_keep = max(1, int(num_heads * (1 - pruning_ratio)))
    
    # Get indices of heads sorted by importance score (descending)
    sorted_indices = torch.argsort(head_importance_scores, descending=True)
    
    # Keep the top-k most important heads
    kept_indices = sorted_indices[:num_heads_to_keep].tolist()
    
    # Sort kept indices in ascending order for deterministic output
    kept_indices.sort()
    
    # Prune attention weights: select only the kept heads
    pruned_weights = attention_weights[kept_indices]
    
    return pruned_weights, kept_indices