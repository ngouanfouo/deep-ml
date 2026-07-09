import numpy as np

def speculative_decode(draft_probs: list, target_probs: list, draft_tokens: list, accept_rand: list, sample_rand: list) -> list:
    """
    Simulate speculative decoding end-to-end.
    
    Args:
        draft_probs: K arrays of shape (vocab_size,) - draft model probability distributions
        target_probs: K+1 arrays of shape (vocab_size,) - target model probability distributions
        draft_tokens: K integers - token indices proposed by draft model
        accept_rand: K floats in [0,1) - random values for acceptance decisions
        sample_rand: K+1 floats in [0,1) - random values for token sampling
    
    Returns:
        List of token indices produced by speculative decoding
    """
    # Convert structures to numpy arrays for reliable element-wise operations
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    
    K = len(draft_tokens)
    final_tokens = []
    all_accepted = True
    
    for i in range(K):
        token = draft_tokens[i]
        q = draft_probs[i, token]
        p = target_probs[i, token]
        
        # Calculate the acceptance probability
        alpha = min(1.0, p / q)
        
        # Check acceptance criteria using the corresponding uniform random value
        if accept_rand[i] < alpha:
            final_tokens.append(token)
        else:
            # Token rejected: calculate the adjusted distribution
            adjusted_dist = np.maximum(0, target_probs[i] - draft_probs[i])
            
            # Normalize the distribution to sum to 1.0
            normalized_dist = adjusted_dist / np.sum(adjusted_dist)
            
            # Inverse CDF sampling using sample_rand[i]
            cumsum = np.cumsum(normalized_dist)
            sampled_token = int(np.where(cumsum >= sample_rand[i])[0][0])
            
            final_tokens.append(sampled_token)
            all_accepted = False
            break
            
    # If all K tokens are accepted, sample a bonus token from target_probs[K]
    if all_accepted:
        cumsum_bonus = np.cumsum(target_probs[K])
        bonus_token = int(np.where(cumsum_bonus >= sample_rand[K])[0][0])
        final_tokens.append(bonus_token)
        
    return final_tokens