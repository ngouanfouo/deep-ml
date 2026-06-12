import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List or array of likelihood ratios (pi_theta / pi_theta_old).
        A: List or array of advantage estimates.
        pi_theta_old: List or array of old policy probabilities (per-sample).
        pi_theta_ref: List or array of reference policy probabilities (per-sample).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value as a float.
    """
    # Convert inputs to numpy arrays for efficient vectorized math
    rhos = np.array(rhos, dtype=float)
    A = np.array(A, dtype=float)
    pi_theta_old = np.array(pi_theta_old, dtype=float)
    pi_theta_ref = np.array(pi_theta_ref, dtype=float)
    
    # 1. Compute the clipped PPO surrogate objective
    clipped_rhos = np.clip(rhos, 1.0 - epsilon, 1.0 + epsilon)
    surrogate_unclipped = rhos * A
    surrogate_clipped = clipped_rhos * A
    l_clip = np.minimum(surrogate_unclipped, surrogate_clipped)
    
    # 2. Recover current policy probability: pi_theta = rho * pi_theta_old
    pi_theta = rhos * pi_theta_old
    
    # 3. Compute the importance-weighted KL divergence
    # r = pi_ref / pi_theta
    r = pi_ref_over_theta = pi_theta_ref / pi_theta
    d_kl = rhos * (r - np.log(r) - 1.0)
    
    # 4. Combine terms and take the group average
    sample_objectives = l_clip - beta * d_kl
    return float(np.mean(sample_objectives))

