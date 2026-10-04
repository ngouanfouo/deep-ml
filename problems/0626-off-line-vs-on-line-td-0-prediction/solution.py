def td_prediction(episodes: list, n_states: int, gamma: float, alpha: float, mode: str) -> list:
    """
    Perform TD(0) prediction in online or offline mode.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        mode: 'online' or 'offline'
    
    Returns:
        List of estimated state values rounded to 4 decimal places
    """
    V = [0.0] * n_states
    
    for episode in episodes:
        if mode == "online":
            for state, reward, next_state, done in episode:
                if done:
                    target = reward
                else:
                    target = reward + gamma * V[next_state]
                V[state] += alpha * (target - V[state])
        
        elif mode == "offline":
            # Snapshot V at the start of the episode
            V_start = V.copy()
            # Accumulate changes per state
            deltas = [0.0] * n_states
            
            for state, reward, next_state, done in episode:
                if done:
                    target = reward
                else:
                    target = reward + gamma * V_start[next_state]
                deltas[state] += alpha * (target - V_start[state])
            
            # Apply all accumulated changes after the episode
            for s in range(n_states):
                V[s] += deltas[s]
        
        else:
            raise ValueError("mode must be 'online' or 'offline'")
    
    return [round(v, 4) for v in V]