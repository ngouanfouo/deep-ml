import numpy as np

def kanerva_coding_td(prototypes: np.ndarray, threshold: float, episodes: list,
                      gamma: float, alpha: float, query_states: np.ndarray) -> tuple:
    """
    Kanerva coding with semi-gradient TD(0) for value function approximation.
    
    Args:
        prototypes: Array of shape (k, d) - prototype points
        threshold: Euclidean distance threshold for activation
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        gamma: Discount factor
        alpha: Learning rate
        query_states: Array of shape (m, d) - states to evaluate
    
    Returns:
        Tuple of (weights, values) as lists of floats rounded to 4 decimal places
    """
    k = prototypes.shape[0]  # Number of prototypes
    
    # Initialize weight vector to zeros
    weights = np.zeros(k)
    
    # Function to compute binary feature vector for a given state
    def get_features(state):
        """Return binary feature vector of length k for a given state."""
        state = np.array(state)
        # Compute Euclidean distances from state to all prototypes
        distances = np.linalg.norm(prototypes - state, axis=1)
        # Activate prototypes within threshold
        features = (distances <= threshold).astype(float)
        return features
    
    # Function to compute value estimate for a given state
    def compute_value(state):
        """Return estimated value for a given state."""
        features = get_features(state)
        return np.dot(weights, features)
    
    # Process all episodes in order
    for episode in episodes:
        for state, reward, next_state, done in episode:
            # Compute features and value for current state
            phi_s = get_features(state)
            V_s = np.dot(weights, phi_s)
            
            # Compute TD target
            if done:
                target = reward
            else:
                # Bootstrap from next state
                V_next = compute_value(next_state)
                target = reward + gamma * V_next
            
            # Compute TD error
            delta = target - V_s
            
            # Update weights using semi-gradient TD(0)
            weights += alpha * delta * phi_s
    
    # Compute values for query states
    values = np.zeros(len(query_states))
    for i, state in enumerate(query_states):
        values[i] = compute_value(state)
    
    # Round to 4 decimal places
    weights_rounded = np.round(weights, 4).tolist()
    values_rounded = np.round(values, 4).tolist()
    
    return (weights_rounded, values_rounded)