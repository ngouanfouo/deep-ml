import numpy as np


def per_decision_importance_sampling(episodes: list,
                                     target_policy: dict,
                                     behavior_policy: dict,
                                     gamma: float) -> tuple:
    """
    Compute per-decision and ordinary importance sampling estimates
    for off-policy evaluation.

    Args:
        episodes: list of episodes, each a list of (state, action, reward) tuples
        target_policy: dict mapping (state, action) -> probability under target policy
        behavior_policy: dict mapping (state, action) -> probability under behavior policy
        gamma: discount factor

    Returns:
        tuple: (pdis_estimate, ois_estimate) both rounded to 4 decimal places
    """
    if not episodes:
        return (0.0, 0.0)

    pdis_returns = []
    ois_returns = []

    for episode in episodes:
        T = len(episode)
        if T == 0:
            pdis_returns.append(0.0)
            ois_returns.append(0.0)
            continue

        # Cumulative importance ratio rho_{0:k} for each step k
        cumulative_rho = 1.0
        pdis_return = 0.0
        discounted_return = 0.0

        for k, (s, a, r) in enumerate(episode):
            ratio = target_policy[(s, a)] / behavior_policy[(s, a)]
            cumulative_rho *= ratio

            # PDIS: each reward weighted by rho up to (and including) step k
            pdis_return += (gamma ** k) * cumulative_rho * r

            # Accumulate discounted return for OIS
            discounted_return += (gamma ** k) * r

        # OIS: full trajectory ratio applied to the full discounted return
        ois_return = cumulative_rho * discounted_return

        pdis_returns.append(pdis_return)
        ois_returns.append(ois_return)

    pdis_estimate = float(np.mean(pdis_returns))
    ois_estimate = float(np.mean(ois_returns))

    return (round(pdis_estimate, 4), round(ois_estimate, 4))