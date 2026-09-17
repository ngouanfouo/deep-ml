import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the discounted return for a given list of rewards.
    Args:
      rewards (list of float): sequence of rewards R_{t+1}, R_{t+2}, ...
      gamma (float): discount factor (0 <= gamma <= 1)
    Returns:
      float: discounted return G_t
    """
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    if n == 0:
        return 0.0

    # gamma^k for k = 0, 1, ..., n-1
    discounts = gamma ** np.arange(n)
    return float(np.sum(discounts * rewards))