import numpy as np

def weighted_importance_sampling(returns: list, target_probs: list, behavior_probs: list) -> tuple:
    """
    Compute ordinary and weighted importance sampling estimates.

    Args:
        returns: List of episode returns (floats), one per episode
        target_probs: List of lists; target_probs[i][t] is the target policy
                      probability for the action taken at step t of episode i
        behavior_probs: List of lists; behavior_probs[i][t] is the behavior policy
                        probability for the action taken at step t of episode i

    Returns:
        Tuple of (ordinary_is_estimate, weighted_is_estimate), each rounded to 4 decimals
    """
    n = len(returns)
    if n == 0:
        return (0.0, 0.0)

    # Compute the importance sampling ratio for each episode
    ratios = []
    for tp, bp in zip(target_probs, behavior_probs):
        r = 1.0
        for p_target, p_behavior in zip(tp, bp):
            r *= p_target / p_behavior
        ratios.append(r)

    ratios = np.asarray(ratios, dtype=float)
    returns = np.asarray(returns, dtype=float)

    # Ordinary importance sampling: simple average of ratio-weighted returns
    ordinary_is = float(np.mean(ratios * returns))

    # Weighted importance sampling: ratio-weighted average
    sum_ratios = float(np.sum(ratios))
    if sum_ratios == 0.0:
        weighted_is = 0.0
    else:
        weighted_is = float(np.sum(ratios * returns) / sum_ratios)

    return (round(ordinary_is, 4), round(weighted_is, 4))