import numpy as np

def discounted_return(rewards, gamma: float) -> float:
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return starting from timestep 0
    """
    # Ensure rewards is a NumPy array
    rewards = np.asarray(rewards, dtype=float)
    
    # Create the powers of gamma: [gamma^0, gamma^1, ..., gamma^(T-1)]
    T = len(rewards)
    powers = gamma ** np.arange(T)
    
    # Compute the sum of the element-wise product
    return float(np.dot(rewards, powers))