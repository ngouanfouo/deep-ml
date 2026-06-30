import numpy as np

def gspo_objective(log_probs_new: list[list[float]], 
                   log_probs_old: list[list[float]], 
                   rewards: list[float], 
                   epsilon: float = 0.2) -> float:
    """
    Compute GSPO (Group Sequence Policy Optimization) clipped objective.
    
    Args:
        log_probs_new: Log probability sequences from new policy
        log_probs_old: Log probability sequences from old policy
        rewards: Reward for each sequence
        epsilon: Clipping range for importance ratios
    
    Returns:
        Average clipped objective value
    """
    # Input validation
    if len(log_probs_new) != len(log_probs_old) or len(log_probs_new) != len(rewards):
        raise ValueError("All input lists must have the same length")
    
    if len(log_probs_new) == 0:
        return 0.0
    
    num_sequences = len(rewards)
    
    # Compute advantages (group-relative rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Avoid division by zero
    if std_reward < 1e-10:
        advantages = np.zeros_like(rewards)
    else:
        advantages = (rewards - mean_reward) / std_reward
    
    # Compute sequence-level importance ratios with length normalization
    sequence_ratios = []
    
    for i in range(num_sequences):
        new_seq = np.array(log_probs_new[i])
        old_seq = np.array(log_probs_old[i])
        
        # Each sequence can have different length
        seq_len = len(new_seq)
        
        # Sum of log probabilities for the sequence
        sum_new = np.sum(new_seq)
        sum_old = np.sum(old_seq)
        
        # Log ratio sum
        log_ratio_sum = sum_new - sum_old
        
        # Length-normalized importance ratio (per-token average)
        # s_i = exp(log_ratio_sum / seq_len)
        ratio = np.exp(log_ratio_sum / seq_len)
        
        sequence_ratios.append(ratio)
    
    sequence_ratios = np.array(sequence_ratios)
    
    # Apply clipping at the sequence level
    clipped_ratios = np.clip(sequence_ratios, 1 - epsilon, 1 + epsilon)
    
    # GSPO objective: E[min(r*A, clip(r)*A)]
    # This is similar to PPO but at the sequence level
    unclipped_objectives = sequence_ratios * advantages
    clipped_objectives = clipped_ratios * advantages
    
    # Take the minimum element-wise
    objectives = np.minimum(unclipped_objectives, clipped_objectives)
    
    # Average over the group
    average_objective = np.mean(objectives)
    
    return float(average_objective)