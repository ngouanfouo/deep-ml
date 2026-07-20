import numpy as np

def certainty_equivalence_td(episodes, n_states, gamma):
    """
    Compute the certainty-equivalence value function from observed transition data.
    
    Args:
        episodes: list of episodes, each episode is a list of (state, reward, next_state) tuples
        n_states: int, number of states in the MDP
        gamma: float, discount factor
    
    Returns:
        np.ndarray: value function of shape (n_states,)
    """
    # Track statistics for transitions, rewards, and source state visits
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    reward_sums = np.zeros(n_states, dtype=np.float64)
    visit_counts = np.zeros(n_states, dtype=np.float64)
    
    # 1. Gather counts from observed episodes
    for episode in episodes:
        for state, reward, next_state in episode:
            visit_counts[state] += 1
            reward_sums[state] += reward
            transition_counts[state, next_state] += 1

    # Identify states that have been visited as a source state at least once
    visited_mask = visit_counts > 0
    
    # Initialize the value function array with zeros
    V = np.zeros(n_states, dtype=np.float64)
    
    # If no states were visited, return early with zeros
    if not np.any(visited_mask):
        return V

    # 2. Build the empirical MDP matrices
    P_hat = np.zeros((n_states, n_states), dtype=np.float64)
    R_hat = np.zeros(n_states, dtype=np.float64)
    
    # Compute empirical expectations safely for visited states
    P_hat[visited_mask] = transition_counts[visited_mask] / visit_counts[visited_mask, np.newaxis]
    R_hat[visited_mask] = reward_sums[visited_mask] / visit_counts[visited_mask]
    
    # 3. Solve the Bellman equation system: (I - gamma * P_hat) @ V = R_hat
    # Create the complete coefficient matrix identity layout
    A = np.eye(n_states) - gamma * P_hat
    
    # Isolate the linear sub-system containing only the active/visited states
    A_sub = A[np.ix_(visited_mask, visited_mask)]
    R_sub = R_hat[visited_mask]
    
    # Directly calculate the exact values for visited positions
    V[visited_mask] = np.linalg.solve(A_sub, R_sub)
        
    return V