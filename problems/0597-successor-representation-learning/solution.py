import numpy as np

def learn_successor_representation(
    experience: list,
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple:
    """
    Learn the Successor Representation from a stream of experience.
    
    Args:
        experience: List of (state, reward, next_state, done) tuples.
        n_states: Number of states in the environment.
        gamma: Discount factor.
        alpha_sr: Learning rate for SR matrix updates.
        alpha_w: Learning rate for reward weight updates.
    
    Returns:
        Tuple of (M, w, V) where:
            M: SR matrix as list of lists, shape (n_states, n_states)
            w: Reward weights as list, length n_states
            V: Value function as list, length n_states
        All values rounded to 4 decimal places.
    """
    # Initialize SR matrix M as zeros (n_states x n_states)
    M = np.zeros((n_states, n_states))
    
    # Initialize reward weight vector w as zeros (length n_states)
    w = np.zeros(n_states)
    
    # Process each experience transition
    for state, reward, next_state, done in experience:
        # 1. Update reward weight for the current state
        # w[state] += alpha_w * (reward - w[state])
        w[state] += alpha_w * (reward - w[state])
        
        # 2. Update SR matrix row for the current state
        # Create one-hot vector for current state
        phi_state = np.zeros(n_states)
        phi_state[state] = 1.0
        
        # Create TD target for SR
        if done:
            # If terminal, no bootstrapping from next state
            target = phi_state
        else:
            # Otherwise, bootstrap from next state: phi_state + gamma * M[next_state, :]
            target = phi_state + gamma * M[next_state, :]
        
        # Update SR row: M[state, :] += alpha_sr * (target - M[state, :])
        M[state, :] += alpha_sr * (target - M[state, :])
    
    # Compute value function: V = M @ w
    V = M @ w
    
    # Round all values to 4 decimal places
    M_rounded = np.round(M, 4).tolist()
    w_rounded = np.round(w, 4).tolist()
    V_rounded = np.round(V, 4).tolist()
    
    return (M_rounded, w_rounded, V_rounded)