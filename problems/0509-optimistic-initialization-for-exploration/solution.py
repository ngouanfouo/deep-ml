def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """
    n_arms = len(true_rewards)

    # Initialize all Q-values optimistically
    Q = [float(initial_q)] * n_arms
    counts = [0] * n_arms

    for _ in range(n_steps):
        # Greedy selection: highest Q-value, tie broken by lowest index
        best = 0
        for i in range(1, n_arms):
            if Q[i] > Q[best]:
                best = i

        # Observe reward and update Q with constant step-size
        reward = true_rewards[best]
        Q[best] += step_size * (reward - Q[best])
        counts[best] += 1

    # Round final Q-values to 4 decimal places
    Q_rounded = [round(q, 4) for q in Q]

    return Q_rounded, counts