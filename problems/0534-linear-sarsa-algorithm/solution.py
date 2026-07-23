import numpy as np

def linear_sarsa(
    episodes: list,
    features: dict,
    n_features: int,
    alpha: float,
    gamma: float
) -> list:
    """
    Episodic semi-gradient Sarsa with linear function approximation.
    
    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        features: Dict mapping (state, action) -> feature vector (list of floats).
        n_features: Number of features.
        alpha: Learning rate.
        gamma: Discount factor.
    
    Returns:
        Final weight vector as a list of floats rounded to 4 decimal places.
    """
    # Initialize weight vector to zeros
    w = np.zeros(n_features)
    
    # Process each episode
    for episode in episodes:
        # Process each step in the episode
        for t in range(len(episode)):
            # Get current transition
            state, action, reward = episode[t]
            
            # Get feature vector for current state-action pair
            x = np.array(features[(state, action)], dtype=float)
            
            # Compute Q(s, a) = w^T * x(s, a)
            q_current = np.dot(w, x)
            
            # Check if this is the terminal step (last transition in episode)
            if t == len(episode) - 1:
                # Terminal state: no next state-action pair
                q_next = 0.0
            else:
                # Get next state and action
                next_state, next_action, _ = episode[t + 1]
                
                # Get feature vector for next state-action pair
                x_next = np.array(features[(next_state, next_action)], dtype=float)
                
                # Compute Q(s', a') = w^T * x(s', a')
                q_next = np.dot(w, x_next)
            
            # Compute TD error: delta = reward + gamma * Q(s', a') - Q(s, a)
            delta = reward + gamma * q_next - q_current
            
            # Update weights: w = w + alpha * delta * x(s, a)
            w = w + alpha * delta * x
    
    # Round to 4 decimal places and convert to list
    return [round(float(val), 4) for val in w]