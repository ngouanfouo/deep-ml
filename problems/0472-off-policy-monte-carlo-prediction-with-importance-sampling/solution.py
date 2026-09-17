import numpy as np

def off_policy_mc_prediction(episodes, target_policy, behavior_policy, gamma, state):
    """
    Estimate state value using off-policy Monte Carlo with importance sampling.
    
    Args:
        episodes: list of episodes, each episode is a list of (state, action, reward) tuples
        target_policy: dict mapping (state, action) -> probability under target policy
        behavior_policy: dict mapping (state, action) -> probability under behavior policy
        gamma: discount factor (float)
        state: the state to evaluate
    
    Returns:
        Tuple of (ordinary_is_estimate, weighted_is_estimate), both rounded to 4 decimals
    """
    ratios = []
    weighted_returns = []

    for episode in episodes:
        # Find the first visit to the target state
        start_idx = None
        for idx, (s, a, r) in enumerate(episode):
            if s == state:
                start_idx = idx
                break
        if start_idx is None:
            continue  # state not visited in this episode

        # Compute the return G from the first visit to the end
        G = 0.0
        discount = 1.0
        for t in range(start_idx, len(episode)):
            _, _, r = episode[t]
            G += discount * r
            discount *= gamma

        # Compute the importance sampling ratio rho
        rho = 1.0
        for t in range(start_idx, len(episode)):
            s_t, a_t, _ = episode[t]
            p_target = target_policy.get((s_t, a_t), 0.0)
            p_behavior = behavior_policy.get((s_t, a_t), 0.0)
            if p_behavior == 0.0:
                # If behavior probability is 0, the episode could not have occurred.
                # In a valid dataset this should not happen; skip to be safe.
                rho = 0.0
                break
            rho *= p_target / p_behavior

        ratios.append(rho)
        weighted_returns.append(rho * G)

    if len(ratios) == 0:
        return (0.0, 0.0)

    ratios = np.array(ratios, dtype=float)
    weighted_returns = np.array(weighted_returns, dtype=float)

    # Ordinary importance sampling: average over episodes that visited the state
    ordinary_is = float(np.mean(weighted_returns))

    # Weighted importance sampling
    sum_ratios = float(np.sum(ratios))
    if sum_ratios == 0.0:
        weighted_is = 0.0
    else:
        weighted_is = float(np.sum(weighted_returns) / sum_ratios)

    return (round(ordinary_is, 4), round(weighted_is, 4))