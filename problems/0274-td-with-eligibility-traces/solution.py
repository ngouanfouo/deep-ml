import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(λ) with accumulating eligibility traces.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        lambd: Trace decay parameter (λ). Use 'lambd' to avoid Python keyword.
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Initialize state values to 0
    V = np.zeros(n_states)
    
    # Iterate over episodes
    for episode in episodes:
        # Initialize eligibility traces
        e = np.zeros(n_states)
        
        # Iterate through each step in the episode
        for t in range(len(episode)):
            state, reward = episode[t]
            
            # If this is the last step, next state is terminal (None)
            if t == len(episode) - 1:
                next_state = None
            else:
                next_state = episode[t + 1][0]
            
            # Compute TD error: δ = R + γ * V(S') - V(S)
            # For terminal state, V(S') = 0
            if next_state is not None:
                delta = reward + gamma * V[next_state] - V[state]
            else:
                delta = reward - V[state]
            
            # Update eligibility traces: e(s) = γλ * e(s) for all s
            e *= gamma * lambd
            
            # Then increment trace for current state: e(S) += 1
            e[state] += 1
            
            # Update value function for all states: V(s) += α * δ * e(s)
            V += alpha * delta * e
    
    return V