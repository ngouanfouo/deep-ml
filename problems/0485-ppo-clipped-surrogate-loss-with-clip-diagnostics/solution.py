import numpy as np

def ppo_clip_loss(old_log_probs: np.ndarray, new_log_probs: np.ndarray, advantages: np.ndarray, epsilon: float = 0.2) -> tuple:
    """
    Compute the PPO clipped surrogate loss and clip fraction diagnostic.
    
    Args:
        old_log_probs: Log-probabilities of actions under the old policy, shape (N,)
        new_log_probs: Log-probabilities of actions under the new policy, shape (N,)
        advantages: Advantage estimates for each sample, shape (N,)
        epsilon: Clipping parameter for the probability ratio
    
    Returns:
        Tuple of (loss, clip_fraction), both rounded to 4 decimal places
        - loss: Mean negated clipped surrogate objective (for minimization)
        - clip_fraction: Fraction of samples where the ratio was clipped
    """
    # 1. Compute probability ratios: r_t = exp(new_log_prob - old_log_prob)
    ratios = np.exp(new_log_probs - old_log_probs)
    
    # 2. Track which samples fall outside the clipping range [1 - epsilon, 1 + epsilon]
    clipped = (ratios < 1.0 - epsilon) | (ratios > 1.0 + epsilon)
    clip_fraction = float(np.mean(clipped))
    
    # 3. Unclipped and clipped surrogate objectives
    surr1 = ratios * advantages
    surr2 = np.clip(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # 4. Take the minimum of the two objectives element-wise, then average over the batch
    # PPO maximizes the surrogate objective, so the loss to minimize is the negative mean.
    surjective_objective = np.minimum(surr1, surr2)
    loss = -np.mean(surjective_objective)
    
    # 5. Round results to 4 decimal places
    return round(float(loss), 4), round(clip_fraction, 4)