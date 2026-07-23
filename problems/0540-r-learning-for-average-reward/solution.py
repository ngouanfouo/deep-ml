import numpy as np

def r_learning(transitions: list, n_states: int, n_actions: int, alpha: float, beta: float, initial_rho: float = 0.0) -> tuple:
    """
    R-Learning algorithm for average reward MDPs.
    
    Args:
        transitions: List of (state, action, reward, next_state) tuples.
        n_states: Number of states.
        n_actions: Number of actions.
        alpha: Learning rate for Q-values.
        beta: Learning rate for average reward estimate.
        initial_rho: Initial average reward estimate.
    
    Returns:
        Tuple of (Q, rho) where Q is a nested list and rho is a float,
        both rounded to 4 decimal places.
    """
    # Initialize Q-values to zeros
    Q = np.zeros((n_states, n_actions))
    
    # Initialize average reward estimate
    rho = initial_rho
    
    # Process each transition
    for s, a, r, s_next in transitions:
        # Compute max_a' Q(s', a')
        max_Q_next = np.max(Q[s_next])
        
        # Compute differential TD error
        # delta = r - rho + max_a' Q(s', a') - Q(s, a)
        delta = r - rho + max_Q_next - Q[s, a]
        
        # Update Q(s, a)
        Q[s, a] += alpha * delta
        
        # Determine if action a is greedy for state s
        # Find the greedy action (ties broken by lowest index)
        greedy_action = np.argmax(Q[s])
        
        # If a is the greedy action, update rho
        if a == greedy_action:
            rho += beta * delta
    
    # Convert Q to nested list with rounding
    Q_rounded = [[round(float(val), 4) for val in row] for row in Q]
    
    # Round rho
    rho_rounded = round(float(rho), 4)
    
    return Q_rounded, rho_rounded