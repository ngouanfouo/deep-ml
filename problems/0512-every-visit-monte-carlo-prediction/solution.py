import numpy as np

def every_visit_mc_prediction(
    episodes: list,
    n_states: int,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using the every-visit Monte Carlo method.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received after leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Accumulators for returns and visit counts
    returns_sum = np.zeros(n_states, dtype=float)
    returns_count = np.zeros(n_states, dtype=float)

    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue

        # Compute the return G_t for every timestep t by working backwards.
        G = 0.0
        G_values = [0.0] * T
        for t in reversed(range(T)):
            _, reward = episode[t]
            G = reward + gamma * G
            G_values[t] = G

        # Every visit to a state contributes its return
        for t in range(T):
            state, _ = episode[t]
            returns_sum[state] += G_values[t]
            returns_count[state] += 1

    # Average returns; unvisited states remain 0.0
    V = np.zeros(n_states, dtype=float)
    visited = returns_count > 0
    V[visited] = returns_sum[visited] / returns_count[visited]

    return V