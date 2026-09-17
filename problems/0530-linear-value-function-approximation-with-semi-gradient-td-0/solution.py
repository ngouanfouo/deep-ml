import numpy as np

def linear_td0_approximation(
    episodes: list,
    n_features: int,
    gamma: float,
    alpha: float
) -> np.ndarray:
    """
    Perform semi-gradient TD(0) with linear function approximation.
    
    Args:
        episodes: List of episodes. Each episode is a list of
                  (state_features, reward, next_state_features, done) tuples.
        n_features: Dimensionality of the feature vectors.
        gamma: Discount factor.
        alpha: Learning rate.
    
    Returns:
        Learned weight vector as numpy array of shape (n_features,).
    """
    # Initialize weights to zeros
    w = np.zeros(n_features, dtype=float)

    for episode in episodes:
        for state_features, reward, next_state_features, done in episode:
            state_features = np.asarray(state_features, dtype=float)
            next_state_features = np.asarray(next_state_features, dtype=float)

            # Current value estimate
            v_current = float(np.dot(w, state_features))

            # Next-state value (zero if terminal)
            if done:
                v_next = 0.0
            else:
                v_next = float(np.dot(w, next_state_features))

            # TD target and TD error
            td_target = reward + gamma * v_next
            td_error = td_target - v_current

            # Semi-gradient update: gradient of v_current w.r.t. w is state_features
            w += alpha * td_error * state_features

    return w