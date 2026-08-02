import numpy as np

def off_policy_control_variate(
    rewards: list,
    values: list,
    target_probs: list,
    behavior_probs: list,
    dones: list,
    gamma: float
) -> tuple:
    """
    Compute off-policy return estimates using per-decision importance
    sampling with a control variate baseline.

    Args:
        rewards: List of T rewards
        values: List of T+1 state-value estimates (includes bootstrap value)
        target_probs: List of T action probabilities under target policy
        behavior_probs: List of T action probabilities under behavior policy
        dones: List of T done flags (1=terminal, 0=non-terminal)
        gamma: Discount factor

    Returns:
        Tuple of (cv_returns, advantages) as lists rounded to 4 decimals
    """
    T = len(rewards)
    cv_returns = [0.0] * T
    advantages = [0.0] * T
    
    # Pre-calculate the importance sampling ratios for each step
    rhos = [t_prob / b_prob for t_prob, b_prob in zip(target_probs, behavior_probs)]
    
    # Initialize future return with the final terminal/bootstrap value vector node
    # However, if the final transition is terminal, it won't propagate across the boundary.
    G = float(values[-1])
    
    # Process the trajectory backwards from T-1 down to 0
    for t in range(T - 1, -1, -1):
        # Check if the current transition ends the episode
        is_terminal = dones[t]
        
        # Calculate the uncorrected full target reward backup from the next state perspective
        # if terminal, the future return propagation across the boundary is cut
        next_return = 0.0 if is_terminal else G
        importance_target = rewards[t] + gamma * next_return
        
        # Apply the control variate blend equation
        G = rhos[t] * importance_target + (1.0 - rhos[t]) * values[t]
        
        cv_returns[t] = G
        advantages[t] = G - values[t]
        
    # Round metrics to 4 decimal places before returning
    cv_returns_rounded = [round(x, 4) for x in cv_returns]
    advantages_rounded = [round(x, 4) for x in advantages]
    
    return cv_returns_rounded, advantages_rounded