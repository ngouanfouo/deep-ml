import numpy as np

def bellman_error_minimization(
    features: np.ndarray,
    transition_probs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    state_distribution: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    initial_weights: np.ndarray = None
) -> tuple:
    """
    Minimize the Mean Squared Bellman Error using gradient descent
    with linear function approximation.
    
    Args:
        features: (n_states, n_features) feature matrix
        transition_probs: (n_states, n_states) transition matrix under the policy
        rewards: (n_states,) expected rewards under the policy
        gamma: discount factor
        state_distribution: (n_states,) weighting distribution over states
        learning_rate: gradient descent step size
        n_iterations: number of gradient descent updates
        initial_weights: (n_features,) initial weight vector, defaults to zeros
    
    Returns:
        Tuple of (weights, msbe, values)
    """
    n_states, n_features = features.shape
    
    # Initialize weights
    if initial_weights is None:
        w = np.zeros(n_features)
    else:
        w = initial_weights.copy()
    
    # Precompute Bellman operator components
    # For linear function approximation: V(s) = features[s] @ w
    # Bellman operator: (T V)(s) = rewards[s] + gamma * sum_j transition_probs[s,j] * V(j)
    
    # Compute expected next-state features under the transition dynamics
    # For each state s: E[features[next_state]] = sum_j P(s,j) * features[j]
    expected_next_features = transition_probs @ features  # Shape: (n_states, n_features)
    
    # Compute Bellman operator in terms of features
    # (T V)(s) = rewards[s] + gamma * expected_next_features[s] @ w
    
    # Precompute the matrix A and vector b for the MSBE gradient
    # MSBE(w) = sum_s d(s) * (features[s] @ w - (rewards[s] + gamma * expected_next_features[s] @ w))^2
    # MSBE(w) = sum_s d(s) * ((features[s] - gamma * expected_next_features[s]) @ w - rewards[s])^2
    
    # Compute the coefficient vector for each state
    coeffs = features - gamma * expected_next_features  # Shape: (n_states, n_features)
    
    # Gradient of MSBE: grad = 2 * sum_s d(s) * (coeffs[s] @ w - rewards[s]) * coeffs[s]
    # We can compute this efficiently using matrix operations
    
    # Weighted coefficient matrix
    weighted_coeffs = state_distribution.reshape(-1, 1) * coeffs  # Shape: (n_states, n_features)
    
    # Perform gradient descent
    for _ in range(n_iterations):
        # Compute Bellman errors for all states
        # V(s) = features[s] @ w
        V = features @ w  # Shape: (n_states,)
        
        # Bellman target: (T V)(s) = rewards[s] + gamma * expected_next_features[s] @ w
        bellman_target = rewards + gamma * (expected_next_features @ w)  # Shape: (n_states,)
        
        # Bellman error: V(s) - (T V)(s)
        bellman_error = V - bellman_target  # Shape: (n_states,)
        
        # Compute gradient of MSBE
        # grad = 2 * sum_s d(s) * bellman_error[s] * (features[s] - gamma * expected_next_features[s])
        grad = 2 * np.sum(state_distribution.reshape(-1, 1) * bellman_error.reshape(-1, 1) * coeffs, axis=0)
        
        # Update weights
        w = w - learning_rate * grad
    
    # Compute final MSBE
    V_final = features @ w
    bellman_target_final = rewards + gamma * (expected_next_features @ w)
    bellman_error_final = V_final - bellman_target_final
    msbe = np.sum(state_distribution * bellman_error_final**2)
    
    # Round to 4 decimal places
    weights_rounded = [round(float(w_i), 4) for w_i in w]
    msbe_rounded = round(float(msbe), 4)
    values_rounded = [round(float(v), 4) for v in V_final]
    
    return (weights_rounded, msbe_rounded, values_rounded)