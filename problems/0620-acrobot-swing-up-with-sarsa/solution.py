import numpy as np

def sarsa_lambda_acrobot(
    features: list,
    rewards: list,
    num_weights: int,
    alpha: float,
    gamma: float,
    lam: float,
    initial_weights: np.ndarray,
    trace_type: str = "accumulating"
) -> list:
    """
    Run one episode of Sarsa(lambda) with linear function approximation.
    
    Args:
        features: List of binary feature vectors x(S_t, A_t) for t=0,...,T-1
        rewards: List of rewards R_{t+1} for t=0,...,T-1
        num_weights: Dimension of weight vector
        alpha: Learning rate
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
        initial_weights: Initial weight vector (numpy array)
        trace_type: 'accumulating' or 'replacing'
    
    Returns:
        Updated weight vector as a list of floats
    """
    # Initialize weights and eligibility traces
    w = initial_weights.copy()
    e = np.zeros(num_weights)
    
    # Number of steps in the episode
    T = len(features)
    
    # Process each step
    for t in range(T):
        # Get current feature vector for state-action pair (S_t, A_t)
        x_t = np.array(features[t], dtype=float)
        
        # Compute Q-value for current state-action pair
        q_t = np.dot(w, x_t)
        
        # Compute Q-value for next state-action pair
        if t == T - 1:
            # Terminal state: next Q-value is 0
            q_next = 0.0
        else:
            # Non-terminal: use next state-action feature vector
            x_next = np.array(features[t + 1], dtype=float)
            q_next = np.dot(w, x_next)
        
        # Compute TD error: delta = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
        delta = rewards[t] + gamma * q_next - q_t
        
        # Update eligibility trace
        if trace_type == "accumulating":
            # Accumulating traces: e = gamma * lambda * e + x
            e = gamma * lam * e + x_t
        elif trace_type == "replacing":
            # Replacing traces: e = 1 for active features, gamma * lambda * e for inactive
            # For binary features, this means:
            # e = gamma * lambda * e for all features
            # Then set e[i] = 1 where x_t[i] == 1
            e = gamma * lam * e
            # For active features (where x_t == 1), set trace to 1
            active_indices = np.where(x_t == 1)[0]
            e[active_indices] = 1.0
        else:
            raise ValueError(f"Unknown trace_type: {trace_type}. Use 'accumulating' or 'replacing'.")
        
        # Update weights: w = w + alpha * delta * e
        w += alpha * delta * e
    
    # Convert to list and round to 4 decimal places for consistency
    return np.round(w, 4).tolist()