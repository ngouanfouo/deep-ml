import numpy as np

def td0_prediction(episodes: list, n_states: int, gamma: float = 0.99, alpha: float = 0.01) -> list:
    """
    Perform TD(0) prediction to estimate the state-value function.
    
    Args:
        episodes: List of episodes, each episode is a list of
                  (state, reward, next_state, done) tuples
        n_states: Total number of states
        gamma: Discount factor
        alpha: Learning rate / step size
    
    Returns:
        List of estimated state values
    """
    # Initialize value function to zeros
    V = np.zeros(n_states, dtype=float)

    for episode in episodes:
        for state, reward, next_state, done in episode:
            # Terminal transitions have no successor value to bootstrap from
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            # TD(0) update
            V[state] += alpha * (td_target - V[state])

    return V.tolist()