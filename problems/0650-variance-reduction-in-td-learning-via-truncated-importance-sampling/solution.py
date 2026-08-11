import numpy as np

def variance_reduced_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    alpha: float,
    gamma: float,
    c_bar: float
) -> dict:
    """
    Off-policy TD(0) prediction with standard and truncated importance sampling.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        alpha: Learning rate.
        gamma: Discount factor.
        c_bar: Truncation threshold for importance sampling ratios.

    Returns:
        Dictionary with V_standard, V_truncated, var_standard, var_truncated.
    """
    # Convert policies to numpy arrays
    behavior = np.array(behavior_policy)
    target = np.array(target_policy)
    
    # Initialize value estimates
    V_standard = np.zeros(num_states)
    V_truncated = np.zeros(num_states)
    
    # Store weighted TD errors for each state
    weighted_errors_standard = [[] for _ in range(num_states)]
    weighted_errors_truncated = [[] for _ in range(num_states)]
    
    # Process each episode
    for episode in episodes:
        T = len(episode)
        
        # Process each transition in the episode
        for t in range(T):
            state, action, reward = episode[t]
            
            # Determine next state value
            if t == T - 1:
                # Terminal transition - next state value is 0
                V_next = 0
            else:
                next_state = episode[t + 1][0]
                V_next_standard = V_standard[next_state]
                V_next_truncated = V_truncated[next_state]
            
            # Compute importance sampling ratio
            # rho = pi(a|s) / b(a|s)
            rho = target[state, action] / behavior[state, action]
            
            # Standard IS TD(0)
            if t == T - 1:
                delta_standard = reward - V_standard[state]
            else:
                delta_standard = reward + gamma * V_standard[next_state] - V_standard[state]
            
            weighted_delta_standard = rho * delta_standard
            V_standard[state] += alpha * weighted_delta_standard
            weighted_errors_standard[state].append(weighted_delta_standard)
            
            # Truncated IS TD(0)
            c = min(rho, c_bar)
            
            if t == T - 1:
                delta_truncated = reward - V_truncated[state]
            else:
                delta_truncated = reward + gamma * V_truncated[next_state] - V_truncated[state]
            
            weighted_delta_truncated = c * delta_truncated
            V_truncated[state] += alpha * weighted_delta_truncated
            weighted_errors_truncated[state].append(weighted_delta_truncated)
    
    # Compute variance for each state
    var_standard = np.zeros(num_states)
    var_truncated = np.zeros(num_states)
    
    for s in range(num_states):
        if len(weighted_errors_standard[s]) > 0:
            var_standard[s] = np.var(weighted_errors_standard[s])
        if len(weighted_errors_truncated[s]) > 0:
            var_truncated[s] = np.var(weighted_errors_truncated[s])
    
    # Round to 4 decimal places
    V_standard = np.round(V_standard, 4)
    V_truncated = np.round(V_truncated, 4)
    var_standard = np.round(var_standard, 4)
    var_truncated = np.round(var_truncated, 4)
    
    return {
        "V_standard": V_standard,
        "V_truncated": V_truncated,
        "var_standard": var_standard,
        "var_truncated": var_truncated
    }