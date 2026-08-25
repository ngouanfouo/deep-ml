import numpy as np

def compute_gae(rewards: list, values: list, dones: list, gamma: float, lam: float) -> tuple:
    """
    Compute GAE advantages and returns for a trajectory with episode boundaries.
    
    Args:
        rewards: List of rewards of length T
        values: List of state values of length T+1 (includes bootstrap value)
        dones: List of done flags of length T (1=terminal, 0=non-terminal)
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        Tuple of (advantages, returns) as lists, each rounded to 4 decimal places
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=float)
    
    T = len(rewards)
    advantages = np.zeros(T, dtype=float)
    gae = 0.0
    
    # Iterate backwards from T-1 down to 0
    for t in reversed(range(T)):
        # If terminal, next_non_terminal is 0, else 1
        # When done=1, the bootstrap value and future GAE propagation are zeroed out.
        next_non_terminal = 1.0 - dones[t]
        next_value = values[t + 1]
        
        # Temporal difference error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE accumulation: gae_t = delta_t + gamma * lam * (1 - done_t) * gae_{t+1}
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        
    # Returns = advantages + values (excluding the final bootstrap value)
    returns = advantages + values[:-1]
    
    # Round to 4 decimal places and convert to lists
    advantages = [round(float(a), 4) for a in advantages]
    returns = [round(float(r), 4) for r in returns]
    
    return advantages, returns