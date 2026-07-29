import numpy as np

def gtd2_prediction(
    features: np.ndarray,
    transitions: list,
    gamma: float,
    alpha_w: float,
    alpha_v: float,
    w_init: np.ndarray,
    v_init: np.ndarray
) -> tuple:
    """
    Perform GTD2 (Gradient TD with second-order correction) for policy evaluation.
    
    Args:
        features: Feature matrix of shape (num_states, d)
        transitions: List of (state, reward, next_state, done) tuples
        gamma: Discount factor
        alpha_w: Learning rate for primary weights
        alpha_v: Learning rate for secondary weights
        w_init: Initial primary weight vector of shape (d,)
        v_init: Initial secondary weight vector of shape (d,)
    
    Returns:
        Tuple of (w, v) as lists of floats rounded to 4 decimal places
    """
    # Convert features to a numpy array just in case it was passed as a list
    features = np.array(features, dtype=float)
    w = np.array(w_init, dtype=float).copy()
    v = np.array(v_init, dtype=float).copy()
    
    for s, r, s_next, done in transitions:
        x = features[s]
        
        if done:
            x_next = np.zeros_like(x)
        else:
            x_next = features[s_next]
            
        # 1. Compute the TD error
        delta = r + gamma * np.dot(w, x_next) - np.dot(w, x)
        
        # 2. Compute the inner product needed for primary update
        x_dot_v = np.dot(x, v)
        
        # Cache current w and v to ensure simultaneous update tracking if needed,
        # though GTD2 allows standard sequential or in-place step updates.
        w_grad = (x - gamma * x_next) * x_dot_v
        v_grad = (delta - x_dot_v) * x
        
        # 3. Perform the updates
        w += alpha_w * w_grad
        v += alpha_v * v_grad

    # Convert to list of floats rounded to 4 decimal places
    w_out = [round(float(val), 4) for val in w]
    v_out = [round(float(val), 4) for val in v]
    
    return w_out, v_out