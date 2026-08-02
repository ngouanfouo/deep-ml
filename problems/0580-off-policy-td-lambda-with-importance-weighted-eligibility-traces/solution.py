import numpy as np

def off_policy_td_lambda(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    gamma: float,
    lam: float,
    alpha: float
) -> np.ndarray:
    """
    Off-policy TD(lambda) prediction with importance-weighted eligibility traces.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        gamma: Discount factor.
        lam: Lambda parameter for trace decay.
        alpha: Learning rate.

    Returns:
        V: numpy array of shape (num_states,) with estimated state values.
    """
    # Initialize all state values to zero
    V = np.zeros(num_states, dtype=float)
    
    # Pre-convert policies to numpy arrays for clean indexing lookup
    pi = np.array(target_policy)
    b = np.array(behavior_policy)
    
    for episode in episodes:
        # Reset eligibility traces to zero at the start of each episode
        e = np.zeros(num_states, dtype=float)
        
        for t in range(len(episode)):
            state, action, reward = episode[t]
            
            # Determine next state if it exists, otherwise it is terminal
            if t + 1 < len(episode):
                next_state = episode[t + 1][0]
                v_next = V[next_state]
            else:
                v_next = 0.0
            
            # Compute importance sampling ratio: rho = pi(a|s) / b(a|s)
            rho = pi[state, action] / b[state, action]
            
            # Compute temporal difference error (delta)
            delta = reward + gamma * v_next - V[state]
            
            # Update eligibility traces following the prompt's instruction sequence:
            # 1. Decay the old traces by gamma * lambda
            # 2. Add the state occurrence indicator
            # 3. Scale the entire resulting trace vector by the importance sampling ratio
            e = gamma * lam * e
            e[state] += 1.0
            e = rho * e
            
            # Update all state values using the TD error weighted by the eligibility traces
            V += alpha * delta * e
            
    return V