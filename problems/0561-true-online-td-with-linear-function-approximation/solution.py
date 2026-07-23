import numpy as np

def true_online_td_lambda(episodes, n_features, alpha, gamma, lam):
    """
    True Online TD(lambda) for policy evaluation with linear function approximation.
    
    Args:
        episodes: List of episodes, each a list of (state_features, reward, next_state_features, done)
        n_features: Dimensionality of feature vectors
        alpha: Step size
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
    
    Returns:
        Final weight vector as numpy array of shape (n_features,)
    """
    # Initialize weight vector to zeros
    w = np.zeros(n_features)
    
    # Process each episode
    for episode in episodes:
        # Reset eligibility trace and auxiliary scalar at start of each episode
        z = np.zeros(n_features)
        V_old = 0.0  # V_old is the previous state's value (V(s_t))
        
        # Process each step in the episode
        for t, (state_features, reward, next_state_features, done) in enumerate(episode):
            # Convert features to numpy arrays
            x = np.array(state_features, dtype=float)
            
            # Compute current state value: V(s) = w^T * x(s)
            V = np.dot(w, x)
            
            # Compute next state value
            if done:
                # Terminal state: V(s') = 0
                V_next = 0.0
                x_next = np.zeros(n_features)
            else:
                x_next = np.array(next_state_features, dtype=float)
                V_next = np.dot(w, x_next)
            
            # Compute TD error: delta = R + gamma * V(s') - V(s)
            delta = reward + gamma * V_next - V
            
            # ---- True Online TD(lambda) updates ----
            
            # Store old weight vector before update
            w_old = w.copy()
            
            # Update eligibility trace (dutch trace)
            # z = (gamma * lam) * z + (1 - alpha * gamma * lam * z^T x) * x
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            
            # Update weights
            # w = w + alpha * (delta + V - V_old) * z - alpha * (V - V_old) * x
            w = w + alpha * (delta + V - V_old) * z - alpha * (V - V_old) * x
            
            # Update V_old for next step
            V_old = V
        
    return w