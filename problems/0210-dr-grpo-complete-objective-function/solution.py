import numpy as np

def compute_dr_grpo_objective(log_probs_new: list[list[float]], 
                               log_probs_old: list[list[float]], 
                               rewards: list[float], 
                               epsilon: float = 0.2) -> float:
    """
    Compute the Dr. GRPO (GRPO Done Right) clipped objective.
    
    Args:
        log_probs_new: Log probabilities from new policy π_θ
                      Each response: [log π_θ(o_1|q), log π_θ(o_2|q,o_1), ...]
        log_probs_old: Log probabilities from old policy π_θ_old
        rewards: Rewards R(q, o_i) for each response
        epsilon: Clipping parameter for importance ratios
    
    Returns:
        Dr. GRPO objective value
    """
    # Convert to numpy arrays
    log_probs_new = np.array(log_probs_new, dtype=np.float64)
    log_probs_old = np.array(log_probs_old, dtype=np.float64)
    rewards = np.array(rewards, dtype=np.float64)
    
    # Step 1: Compute advantages (unbiased: reward minus mean)
    mean_reward = np.mean(rewards)
    advantages = rewards - mean_reward  # Shape: (num_responses,)
    
    # Step 2: Compute token-level importance ratios
    # ratio = exp(log π_new - log π_old) for each token
    log_ratios = log_probs_new - log_probs_old
    ratios = np.exp(log_ratios)  # Shape: (num_responses, num_tokens)
    
    # Step 3: Clipped objective with token-level ratios
    # For each token: min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
    clipped_ratios = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    
    # Expand advantages to match token dimension
    advantages_expanded = advantages[:, np.newaxis]  # Shape: (num_responses, 1)
    
    # Token-level objective values
    obj_tokens = np.minimum(
        ratios * advantages_expanded,
        clipped_ratios * advantages_expanded
    )  # Shape: (num_responses, num_tokens)
    
    # Step 4: Sum over tokens for each response, then average over responses
    response_objectives = np.sum(obj_tokens, axis=1)  # Shape: (num_responses,)
    objective = np.mean(response_objectives)
    
    return float(objective)