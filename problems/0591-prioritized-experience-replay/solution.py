import numpy as np

def prioritized_replay_sample(priorities: list, batch_size: int, alpha: float = 0.6, beta: float = 0.4, seed: int = 42) -> dict:
    """
    Sample a batch from a replay buffer using prioritized experience replay.

    Args:
        priorities: list of priority values for each experience (positive floats)
        batch_size: number of experiences to sample
        alpha: prioritization exponent (0 = uniform, 1 = full prioritization)
        beta: importance sampling exponent (0 = no correction, 1 = full correction)
        seed: random seed for reproducibility

    Returns:
        dict with 'indices', 'probabilities', and 'weights'
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)
    
    N = len(priorities)
    priorities_arr = np.array(priorities, dtype=float)
    
    # 1. Compute sampling probabilities
    scaled_priorities = priorities_arr ** alpha
    probabilities = scaled_priorities / np.sum(scaled_priorities)
    
    # 2. Sample batch indices without replacement
    sampled_indices = rng.choice(N, size=batch_size, replace=False, p=probabilities)
    
    # 3. Compute Importance Sampling (IS) weights for sampled experiences
    # w_i = (N * P_i) ^ (-beta)
    sampled_probs = probabilities[sampled_indices]
    weights = (N * sampled_probs) ** (-beta)
    
    # Max-normalize the weights to scale the gradient updates smoothly
    max_weight = np.max(weights)
    if max_weight > 0:
        normalized_weights = weights / max_weight
    else:
        normalized_weights = weights
        
    return {
        'indices': sampled_indices.tolist(),
        'probabilities': np.round(probabilities, 4).tolist(),
        'weights': np.round(normalized_weights, 4).tolist()
    }