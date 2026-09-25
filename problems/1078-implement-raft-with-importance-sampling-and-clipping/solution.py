import numpy as np

def raft_pp_loss(new_logps, old_logps, rewards, mask, clip_eps):
    """Compute the RAFT++ loss with importance-sampling correction and PPO-style clipping.

    Args:
        new_logps: (N, T) per-token log-probs under the current policy.
        old_logps: (N, T) per-token log-probs under the old (rollout) policy.
        rewards:   (N,) binary rewards used as a filter (1 = keep).
        mask:      (N, T) token validity mask.
        clip_eps:  float, PPO clip epsilon.

    Returns:
        float scalar loss.
    """
    new_logps = np.asarray(new_logps, dtype=float)
    old_logps = np.asarray(old_logps, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    mask = np.asarray(mask, dtype=float)

    # Keep only responses that received a positive reward.
    kept = np.where(rewards > 0)[0]
    if kept.size == 0:
        return 0.0

    per_sequence_means = []
    for i in kept:
        # Per-token importance ratio
        ratio = np.exp(new_logps[i] - old_logps[i])

        # PPO-style clipping and surrogate
        clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = np.minimum(ratio, clipped_ratio)

        # Average over valid tokens only
        m = mask[i]
        valid_count = m.sum()
        if valid_count > 0:
            seq_mean = (surrogate * m).sum() / valid_count
        else:
            seq_mean = 0.0
        per_sequence_means.append(seq_mean)

    loss = -float(np.mean(per_sequence_means))
    return loss