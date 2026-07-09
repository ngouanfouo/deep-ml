import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    # Convert inputs to numpy arrays for efficient vector/matrix operations
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    
    accepted_tokens = []
    
    for i, token in enumerate(draft_tokens):
        q = draft_probs[i, token]
        p = target_probs[i, token]
        
        # Calculate the acceptance probability
        alpha = min(1.0, p / q)
        
        if coin_flips[i] < alpha:
            # Token accepted: add to list and proceed to the next position
            accepted_tokens.append(token)
        else:
            # Token rejected: compute the adjusted distribution
            adjusted_dist = np.maximum(0, target_probs[i] - draft_probs[i])
            
            # Normalize the distribution so it sums to 1.0
            normalized_dist = adjusted_dist / np.sum(adjusted_dist)
            
            # Sample a replacement token using cumulative sum and the resample_coin
            cumsum = np.cumsum(normalized_dist)
            rejected_replacement = int(np.where(cumsum >= resample_coin)[0][0])
            
            accepted_tokens.append(rejected_replacement)
            break
            
    return accepted_tokens