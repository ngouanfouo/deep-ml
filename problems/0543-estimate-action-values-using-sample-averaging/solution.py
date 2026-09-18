def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.

    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step

    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    sums = [0.0] * k
    N = [0] * k

    for a, r in zip(actions, rewards):
        sums[a] += float(r)
        N[a] += 1

    Q = [round(sums[i] / N[i], 4) if N[i] > 0 else 0.0 for i in range(k)]

    return Q, N