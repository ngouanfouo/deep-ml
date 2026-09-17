import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)
    """
    Q = np.asarray(Q, dtype=float)

    # Optimal action per state (np.argmax returns the first max index on ties)
    optimal_actions = np.argmax(Q, axis=1).astype(int)

    # Optimal state values V*(s) = max_a Q(s, a)
    state_values = np.max(Q, axis=1)

    # Advantage A(s, a) = Q(s, a) - V*(s)
    advantages = Q - state_values[:, np.newaxis]

    return {
        'optimal_actions': optimal_actions.tolist(),
        'state_values': np.round(state_values, 4).tolist(),
        'advantages': np.round(advantages, 4).tolist()
    }