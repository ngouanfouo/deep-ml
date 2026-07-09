import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Apply a combined sampling pipeline to raw logits.
    
    Args:
        logits: Raw unnormalized scores for each token
        temperature: Scaling factor for logit sharpness (0 = greedy)
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dict with 'probabilities' (list of floats rounded to 4 decimals)
        and 'sampled_token' (int index of chosen token)
    """
    rng = np.random.default_rng(seed)
    logits = np.array(logits, dtype=float)
    vocab_size = len(logits)
    
    # --- 1. Temperature Scaling ---
    if temperature <= 0:
        # Greedy decoding: Deterministically choose the highest logit token
        sampled_token = int(np.argmax(logits))
        probabilities = np.zeros(vocab_size)
        probabilities[sampled_token] = 1.0
        return {
            'probabilities': probabilities.tolist(),
            'sampled_token': sampled_token
        }
    
    logits = logits / temperature

    # --- 2. Top-k Filtering ---
    if 0 < top_k < vocab_size:
        # Find the threshold value of the k-th highest logit
        # Elements smaller than this will be masked out to negative infinity
        kth_largest_val = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth_largest_val] = -np.inf

    # --- 3. Convert to Probabilities (Softmax) ---
    # Subtracting the max for numerical stability against overflow
    max_logit = np.max(logits)
    if max_logit == -np.inf:
        # Guard against edge case where all logits were masked
        probabilities = np.ones(vocab_size) / vocab_size
    else:
        exp_logits = np.exp(logits - max_logit)
        probabilities = exp_logits / np.sum(exp_logits)

    # --- 4. Top-p (Nucleus) Filtering ---
    if top_p < 1.0:
        # Sort indices by probability in descending order
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices]
        
        # Calculate cumulative distribution function
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Keep tokens up to the first one that exceeds or meets top_p
        # shifted by 1 to include the token that crosses the threshold
        keep_mask = cumsum_probs <= top_p
        if not np.any(keep_mask):
            # If even the top 1 token exceeds top_p, keep at least the top 1
            keep_mask[0] = True
        else:
            # Shift threshold boundary outward by one to include the crossing token
            first_crossing_idx = np.where(~keep_mask)[0]
            if len(first_crossing_idx) > 0:
                keep_mask[first_crossing_idx[0]] = True
                
        # Get indices to zero out
        indices_to_remove = sorted_indices[~keep_mask]
        probabilities[indices_to_remove] = 0.0
        
        # Re-normalize remaining distribution safely
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            probabilities = np.ones(vocab_size) / vocab_size

    # --- 5. Sampling ---
    sampled_token = int(rng.choice(vocab_size, p=probabilities))
    
    # Format and round output probabilities
    rounded_probs = np.round(probabilities, 4).tolist()

    return {
        'probabilities': rounded_probs,
        'sampled_token': sampled_token
    }