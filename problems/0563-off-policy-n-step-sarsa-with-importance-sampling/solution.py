import numpy as np

def off_policy_nstep_sarsa(
    states: list,
    actions: list,
    rewards: list,
    Q: dict,
    target_policy: dict,
    behavior_policy: dict,
    n: int,
    gamma: float,
    alpha: float
) -> dict:
    """
    Perform off-policy n-step Sarsa with importance sampling on a single episode.
    
    Args:
        states: List of states visited in the episode (length T+1)
        actions: List of actions taken (length T)
        rewards: List of rewards received (length T)
        Q: Action-value function as dict {(state, action): value}
        target_policy: Target policy {state: {action: prob}}
        behavior_policy: Behavior policy {state: {action: prob}}
        n: Number of steps for the return
        gamma: Discount factor
        alpha: Learning rate
    
    Returns:
        Updated Q dictionary
    """
    # Make a copy of Q to avoid modifying the original
    Q = dict(Q)
    
    # Helper function to get Q value, returning 0.0 if missing
    def get_Q(state, action):
        return Q.get((state, action), 0.0)
    
    T = len(states) - 1  # Number of time steps
    
    # Process each timestep t from 0 to T-1
    for t in range(T):
        # Determine horizon: min(t + n, T)
        h = min(t + n, T)
        
        # Compute n-step return G
        # G = R_{t+1} + gamma * R_{t+2} + ... + gamma^(h-t-1) * R_h
        # + gamma^(h-t) * Q(S_h, A_h) if h < T, else 0
        G = 0.0
        for k in range(t + 1, h + 1):
            G += (gamma ** (k - t - 1)) * rewards[k - 1]  # rewards is 0-indexed: rewards[0] = R_1
        
        # If horizon is not the terminal state, add bootstrap value
        if h < T:
            G += (gamma ** (h - t)) * get_Q(states[h], actions[h])
        
        # Compute importance sampling ratio
        # Product of pi(A_k | S_k) / b(A_k | S_k) for k from t+1 to h-1
        # (exclusive of both endpoints: from the action after A_t to before A_h)
        rho = 1.0
        for k in range(t + 1, h):
            state = states[k]
            action = actions[k]
            target_prob = target_policy.get(state, {}).get(action, 0.0)
            behavior_prob = behavior_policy.get(state, {}).get(action, 0.0)
            
            # Avoid division by zero
            if behavior_prob > 0:
                rho *= target_prob / behavior_prob
            else:
                rho = 0.0
                break
        
        # Update Q(S_t, A_t)
        state = states[t]
        action = actions[t]
        current_Q = get_Q(state, action)
        td_error = G - current_Q
        Q[(state, action)] = current_Q + alpha * rho * td_error
    
    return Q