import numpy as np

def lstd(features, next_features, rewards, gamma, epsilon=0.01):
    """
    Least-Squares Temporal Difference (LSTD) for policy evaluation.
    
    Args:
        features: array-like of shape (T, d) - feature vectors for each state
        next_features: array-like of shape (T, d) - feature vectors for next states
        rewards: array-like of length T - observed rewards
        gamma: float - discount factor
        epsilon: float - regularization constant
    
    Returns:
        List of floats - weight vector w of length d
    """
    # Convert inputs to numpy arrays
    features = np.array(features, dtype=float)
    next_features = np.array(next_features, dtype=float)
    rewards = np.array(rewards, dtype=float)
    
    # Get dimensions
    T, d = features.shape
    
    # Initialize accumulated matrix A and vector b
    A = np.zeros((d, d))
    b = np.zeros(d)
    
    # Accumulate statistics over all transitions
    for t in range(T):
        phi = features[t]           # φ(s_t)
        phi_next = next_features[t]  # φ(s_{t+1})
        r = rewards[t]               # r_t
        
        # A += φ(s_t) * (φ(s_t) - γ * φ(s_{t+1}))^T
        A += np.outer(phi, phi - gamma * phi_next)
        
        # b += φ(s_t) * r_t
        b += phi * r
    
    # Add regularization to the diagonal of A
    A += epsilon * np.eye(d)
    
    # Solve the linear system A * w = b
    w = np.linalg.solve(A, b)
    
    # Round to 4 decimal places and convert to list
    return [round(float(val), 4) for val in w]