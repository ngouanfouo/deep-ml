import numpy as np

def residual_gradient_td(episodes, features, gamma, alpha, n_passes):
    """
    Run the Residual Gradient algorithm for policy evaluation
    with linear function approximation.
    
    Args:
        episodes: list of episodes, each is a list of (state, reward, next_state) tuples.
                  next_state = -1 indicates terminal (value 0).
        features: np.ndarray of shape (n_states, d), feature vectors per state.
        gamma: float, discount factor.
        alpha: float, step size.
        n_passes: int, number of passes through the data.
    
    Returns:
        np.ndarray of shape (d,): learned weight vector.
    """
    # Initialize weight vector to zeros
    d = features.shape[1]
    w = np.zeros(d)
    
    # Pre-compute features for terminal state (zero vector)
    terminal_features = np.zeros(d)
    
    for _ in range(n_passes):
        for episode in episodes:
            for state, reward, next_state in episode:
                # Get feature vectors
                phi_s = features[state] if state >= 0 else terminal_features
                
                if next_state == -1:
                    phi_next = terminal_features
                    # Terminal state value is 0
                    V_next = 0
                else:
                    phi_next = features[next_state]
                    V_next = np.dot(phi_next, w)
                
                # Current value estimate
                V_s = np.dot(phi_s, w)
                
                # TD error (the full TD error, not just delta for gradient)
                # δ = R + γ * V(s') - V(s)
                delta = reward + gamma * V_next - V_s
                
                # Residual gradient update:
                # w ← w + α * δ * (∇_w V(s) - γ * ∇_w V(s'))
                # For linear approximation: ∇_w V(s) = φ(s), ∇_w V(s') = φ(s')
                # But we need to use the original features, not their values
                # The gradient direction is: φ(s) - γ * φ(s')
                grad = phi_s - gamma * phi_next
                
                # Update weights
                w += alpha * delta * grad
    
    return w