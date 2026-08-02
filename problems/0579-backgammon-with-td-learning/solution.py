import numpy as np

def td_gammon_learning(game_states, outcome, W1, b1, w2, b2, alpha, lambd):
    """
    Apply TD(lambda) learning to update a neural network value function
    over a single self-play game sequence.
    
    Args:
        game_states: list of numpy arrays, board features at each time step
        outcome: float, final game result (1.0 = win, 0.0 = loss)
        W1: np.ndarray of shape (n_hidden, n_features), hidden weights
        b1: np.ndarray of shape (n_hidden,), hidden biases
        w2: np.ndarray of shape (n_hidden,), output weights
        b2: float, output bias
        alpha: float, learning rate
        lambd: float, eligibility trace decay (lambda)
    
    Returns:
        Tuple of (W1, b1, w2, b2) after all TD updates
    """
    # Helper function for sigmoid with numerical stability
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    # Make copies of parameters to avoid modifying inputs
    W1 = W1.copy()
    b1 = b1.copy()
    w2 = w2.copy()
    b2 = float(b2)
    
    # Initialize eligibility traces
    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_w2 = np.zeros_like(w2)
    e_b2 = 0.0
    
    T = len(game_states)
    
    # Process each time step sequentially
    for t in range(T):
        state_t = game_states[t]
        
        # Compute V(s_t) and h_t using current weights at time step t
        z1_t = W1 @ state_t + b1
        h_t = sigmoid(z1_t)
        z2_t = np.dot(w2, h_t) + b2
        V_t = sigmoid(z2_t)
        
        # Compute TD error delta
        if t < T - 1:
            # Compute V(s_{t+1}) using the SAME weights as V(s_t)
            state_next = game_states[t + 1]
            z1_next = W1 @ state_next + b1
            h_next = sigmoid(z1_next)
            z2_next = np.dot(w2, h_next) + b2
            V_next = sigmoid(z2_next)
            delta = V_next - V_t
        else:
            # Final step uses the absolute game outcome
            delta = outcome - V_t
            
        # Compute gradients with respect to V_t using backpropagation
        grad_V = V_t * (1.0 - V_t)
        
        # Output layer gradients
        grad_b2 = grad_V
        grad_w2 = grad_V * h_t
        
        # Hidden layer gradients
        grad_h = grad_V * w2
        grad_z1 = grad_h * (h_t * (1.0 - h_t))
        grad_b1 = grad_z1
        grad_W1 = np.outer(grad_z1, state_t)
        
        # Accumulate eligibility traces
        e_W1 = lambd * e_W1 + grad_W1
        e_b1 = lambd * e_b1 + grad_b1
        e_w2 = lambd * e_w2 + grad_w2
        e_b2 = lambd * e_b2 + grad_b2
        
        # Apply weight updates at the end of step t
        W1 += alpha * delta * e_W1
        b1 += alpha * delta * e_b1
        w2 += alpha * delta * e_w2
        b2 += alpha * delta * e_b2
        
    return W1, b1, w2, b2