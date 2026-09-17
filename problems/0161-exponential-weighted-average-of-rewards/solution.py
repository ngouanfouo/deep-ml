def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k = len(rewards)
    result = ((1.0 - alpha) ** k) * Q1

    for i, R in enumerate(rewards, start=1):
        result += alpha * ((1.0 - alpha) ** (k - i)) * R

    return result