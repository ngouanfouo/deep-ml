import torch

def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a torch.Tensor
    """
    # Convert logits to tensor
    logits_tensor = torch.tensor(logits)
    
    # Compute softmax probabilities
    probs = torch.softmax(logits_tensor, dim=0)
    
    # Get indices sorted by probability in descending order
    # For tie-breaking, we want smaller indices first when probabilities are equal
    # We can achieve this by sorting by (-prob, index)
    sorted_indices = torch.argsort(probs, descending=True)
    
    # Sort probabilities according to sorted indices
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=0)
    
    # Find the cutoff index: first position where cumsum >= p
    # Add 1 to include the token that pushes cumulative sum over the threshold
    cutoff_idx = torch.searchsorted(cumsum, p) + 1
    
    # Create mask: tokens in the nucleus (first cutoff_idx sorted tokens)
    mask = torch.zeros_like(probs)
    mask[sorted_indices[:cutoff_idx]] = 1.0
    
    # Apply mask to probabilities (zero out tokens outside nucleus)
    filtered_probs = probs * mask
    
    # Renormalize (sum to 1)
    normalized_probs = filtered_probs / filtered_probs.sum()
    
    return normalized_probs