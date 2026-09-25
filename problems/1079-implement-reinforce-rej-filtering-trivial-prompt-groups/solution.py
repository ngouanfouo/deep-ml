import numpy as np

def reinforce_rej_loss(log_probs: np.ndarray, rewards: np.ndarray) -> float:
    """
    Compute the Reinforce-Rej loss by filtering out trivial prompt groups
    (all-correct or all-incorrect) and applying REINFORCE on the rest.

    Args:
        log_probs: (P, N) array of per-response log-probabilities.
        rewards:   (P, N) array of binary rewards in {0, 1}.

    Returns:
        Scalar loss as a Python float.
    """
    log_probs = np.asarray(log_probs, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    P, N = rewards.shape

    # Identify non-trivial groups: not all-0 and not all-1.
    row_sums = rewards.sum(axis=1)
    nontrivial = (row_sums > 0) & (row_sums < N)

    if not np.any(nontrivial):
        return 0.0

    # Gather all entries from surviving groups (all N samples per surviving prompt).
    surviving_log_probs = log_probs[nontrivial]   # (P', N)
    surviving_rewards = rewards[nontrivial]       # (P', N)

    # REINFORCE: negative mean of log_prob * reward over all surviving entries.
    products = surviving_log_probs * surviving_rewards
    loss = -float(products.mean())
    return loss