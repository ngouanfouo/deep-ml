import numpy as np

def off_policy_mc_control(episodes: list, behavior_policy: dict, n_states: int, n_actions: int, gamma: float = 1.0) -> list:
    """
    Off-policy Monte Carlo control using weighted importance sampling.
    """
    # Initialize Q and C to zeros
    Q = np.zeros((n_states, n_actions), dtype=float)
    C = np.zeros((n_states, n_actions), dtype=float)
    
    for episode in episodes:
        # Episode is a list of (state, action, reward) tuples
        # Process episode backwards
        G = 0.0
        W = 1.0
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # Update return
            G = gamma * G + reward
            
            # Update C(s, a)
            C[state, action] += W
            
            # Update Q(s, a) using weighted average
            Q[state, action] += (W / C[state, action]) * (G - Q[state, action])
            
            # Get greedy action for this state (break ties by lowest index)
            greedy_action = np.argmax(Q[state])
            
            # If the action taken doesn't match the greedy action, stop processing
            if action != greedy_action:
                break
            
            # Update importance sampling weight
            # W = W * (1 / behavior_policy(state, action))
            W = W / behavior_policy.get((state, action), 1.0)
    
    # Round to 4 decimal places and convert to list
    Q_rounded = np.round(Q, 4).tolist()
    
    return Q_rounded