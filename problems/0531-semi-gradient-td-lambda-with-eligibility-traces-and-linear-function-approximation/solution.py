import numpy as np

def semi_gradient_td_lambda(episodes: list, n_features: int, alpha: float, gamma: float, lam: float, initial_w: list = None) -> list:
    """
    Implement semi-gradient TD(lambda) with accumulating eligibility traces
    and linear function approximation.
    
    Args:
        episodes: List of episodes. Each episode is a list of tuples
                  (state_features, reward, next_state_features).
                  next_state_features is None for terminal transitions.
        n_features: Dimensionality of feature vectors
        alpha: Learning rate
        gamma: Discount factor
        lam: Lambda parameter for eligibility trace decay
        initial_w: Optional initial weight vector (defaults to zeros)
    
    Returns:
        Final weight vector as a list of floats rounded to 4 decimal places.
    """
    # Initialize weights
    if initial_w is None:
        w = np.zeros(n_features)
    else:
        w = np.array(initial_w, dtype=float)
    
    # Process each episode
    for episode in episodes:
        # Reset eligibility trace at the start of each episode
        z = np.zeros(n_features)
        
        # Process each transition in the episode
        for state_features, reward, next_state_features in episode:
            # Convert features to numpy arrays
            x = np.array(state_features, dtype=float)
            
            # Compute value of current state: V(s) = w^T * x
            V_s = np.dot(w, x)
            
            # Compute value of next state: V(s') = w^T * x'
            # If next_state is terminal, V(s') = 0
            if next_state_features is None:
                V_s_next = 0.0
            else:
                x_next = np.array(next_state_features, dtype=float)
                V_s_next = np.dot(w, x_next)
            
            # Compute TD error: delta = reward + gamma * V(s') - V(s)
            delta = reward + gamma * V_s_next - V_s
            
            # Update eligibility trace: z = gamma * lambda * z + x
            z = gamma * lam * z + x
            
            # Update weights: w = w + alpha * delta * z
            w = w + alpha * delta * z
    
    # Round to 4 decimal places and convert to list
    return [round(float(val), 4) for val in w]