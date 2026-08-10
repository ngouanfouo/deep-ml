import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1 - s)

def td_position_eval(
    states: list,
    outcome: float,
    W1: list,
    b1: list,
    W2: list,
    b2: list,
    alpha: float,
    gamma: float,
    lam: float
) -> dict:
    """
    Train a position evaluation network using online TD(lambda).

    Args:
        states: List of feature vectors, each a list of floats.
        outcome: Terminal game outcome (float).
        W1: Hidden weights, shape (n_hidden, n_features).
        b1: Hidden biases, shape (n_hidden,).
        W2: Output weights, shape (1, n_hidden).
        b2: Output bias, shape (1,).
        alpha: Learning rate.
        gamma: Discount factor.
        lam: Eligibility trace decay parameter.

    Returns:
        Dict with 'W1', 'b1', 'W2', 'b2' as lists, rounded to 4 decimals.
    """
    # Convert inputs to numpy arrays
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float).flatten()  # Make it 1D for easier handling
    b2 = np.array(b2, dtype=float).flatten()[0]
    
    n_hidden = W1.shape[0]
    n_features = W1.shape[1]
    
    # Initialize eligibility traces (same shapes as parameters)
    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_W2 = np.zeros_like(W2)
    e_b2 = 0.0
    
    # Number of states in the episode
    T = len(states)
    
    # Process each state sequentially
    for t in range(T):
        x = np.array(states[t], dtype=float)
        
        # Forward pass to compute value and gradients
        # Hidden layer
        z1 = W1 @ x + b1
        h = sigmoid(z1)
        
        # Output layer
        z2 = W2 @ h + b2
        v = sigmoid(z2)
        
        # Compute gradients for current state with current weights
        # Output layer gradients
        dv_dz2 = v * (1 - v)  # derivative of sigmoid at z2
        
        # Gradients w.r.t. output parameters
        grad_W2 = dv_dz2 * h  # shape (n_hidden,)
        grad_b2 = dv_dz2
        
        # Hidden layer gradients (backpropagation)
        # dz1 = (dv/dz2) * (dz2/dh) * (dh/dz1)
        # dz2/dh = W2
        # dh/dz1 = h * (1 - h)
        dz1 = dv_dz2 * W2 * sigmoid_derivative(z1)
        grad_W1 = np.outer(dz1, x)  # shape (n_hidden, n_features)
        grad_b1 = dz1
        
        # Update eligibility traces: e = gamma * lambda * e + gradient
        e_W1 = gamma * lam * e_W1 + grad_W1
        e_b1 = gamma * lam * e_b1 + grad_b1
        e_W2 = gamma * lam * e_W2 + grad_W2
        e_b2 = gamma * lam * e_b2 + grad_b2
        
        # Compute TD error
        if t == T - 1:
            # Last state: use the actual outcome as target
            target = outcome
        else:
            # Non-terminal: use value of next state (with current weights)
            x_next = np.array(states[t + 1], dtype=float)
            z1_next = W1 @ x_next + b1
            h_next = sigmoid(z1_next)
            z2_next = W2 @ h_next + b2
            v_next = sigmoid(z2_next)
            target = gamma * v_next
        
        td_error = target - v
        
        # Apply weight updates using eligibility traces
        W1 += alpha * td_error * e_W1
        b1 += alpha * td_error * e_b1
        W2 += alpha * td_error * e_W2
        b2 += alpha * td_error * e_b2
    
    # Convert back to nested lists and round to 4 decimal places
    return {
        'W1': np.round(W1, 4).tolist(),
        'b1': np.round(b1, 4).tolist(),
        'W2': [np.round(W2, 4).tolist()],  # W2 should be (1, n_hidden)
        'b2': [round(float(b2), 4)]  # b2 should be (1,)
    }