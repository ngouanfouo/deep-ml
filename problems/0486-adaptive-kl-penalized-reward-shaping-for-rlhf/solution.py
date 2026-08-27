import numpy as np

def adaptive_kl_penalized_reward(rewards, log_probs_policy, log_probs_reference, beta, kl_target, beta_update_factor=1.5):
    """
    Compute KL-penalized shaped rewards with adaptive beta adjustment.

    Args:
        rewards: list of scalar rewards for each response
        log_probs_policy: list of lists, per-token log-probs under policy for each response
        log_probs_reference: list of lists, per-token log-probs under reference for each response
        beta: KL penalty coefficient
        kl_target: target KL divergence for adaptive adjustment
        beta_update_factor: multiplicative factor for beta adjustment (default 1.5)

    Returns:
        dict with 'per_response_kl', 'mean_kl', 'shaped_rewards', 'updated_beta'
    """
    # Compute per-response KL divergence
    per_response_kl = []
    for policy_probs, ref_probs in zip(log_probs_policy, log_probs_reference):
        # Convert to numpy arrays for efficient computation
        policy_probs = np.array(policy_probs)
        ref_probs = np.array(ref_probs)
        
        # Per-token KL = policy_log_prob - reference_log_prob
        per_token_kl = policy_probs - ref_probs
        
        # Sum across tokens for per-response KL
        response_kl = np.sum(per_token_kl)
        per_response_kl.append(float(response_kl))
    
    # Convert to numpy array for easier computation
    per_response_kl = np.array(per_response_kl)
    
    # Compute mean KL across all responses
    mean_kl = float(np.mean(per_response_kl))
    
    # Compute shaped rewards: reward - beta * KL
    shaped_rewards = []
    for reward, kl in zip(rewards, per_response_kl):
        shaped_reward = reward - beta * kl
        shaped_rewards.append(float(shaped_reward))
    
    # Adaptively update beta based on mean KL
    updated_beta = beta
    if mean_kl > 1.5 * kl_target:
        updated_beta = beta * beta_update_factor
    elif mean_kl < kl_target / 1.5:
        updated_beta = beta / beta_update_factor
    # else beta remains unchanged
    
    # Round all values to 4 decimal places
    return {
        'per_response_kl': [round(float(kl), 4) for kl in per_response_kl],
        'mean_kl': round(mean_kl, 4),
        'shaped_rewards': [round(float(sr), 4) for sr in shaped_rewards],
        'updated_beta': round(float(updated_beta), 4)
    }