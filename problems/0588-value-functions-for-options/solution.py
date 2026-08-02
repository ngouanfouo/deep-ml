import numpy as np

def option_value_iteration(R: np.ndarray, P: np.ndarray, options: list, gamma: float, num_iterations: int) -> dict:
    """
    Compute value functions for options using iterative Bellman updates.
    
    Args:
        R: Reward matrix of shape (S, A)
        P: Transition probability tensor of shape (S, A, S)
        options: List of option dicts with 'policy' (S, A) and 'termination' (S,)
        gamma: Discount factor
        num_iterations: Number of iteration sweeps
    
    Returns:
        Dictionary with 'Q_options', 'V', and 'U'
    """
    S, A = R.shape
    num_options = len(options)
    
    # Initialize the option-value function to zero
    Q = np.zeros((S, num_options), dtype=float)
    
    # Pre-extract policy and termination arrays for speed and clarity
    policies = [np.array(opt['policy']) for opt in options]
    terminations = [np.array(opt['termination']) for opt in options]
    
    V = np.zeros(S, dtype=float)
    U = np.zeros((num_options, S), dtype=float)
    
    for _ in range(num_iterations):
        # 1. Derive the state-value function V(s) from current Q-values
        V = np.max(Q, axis=1)
        
        # 2. Derive the upon-arrival function U(o, s') from current Q and V
        for o in range(num_options):
            beta = terminations[o]
            U[o] = (1.0 - beta) * Q[:, o] + beta * V
            
        # 3. Update the option-value function Q(s, o) using the option Bellman equation
        Q_next = np.zeros_like(Q)
        for o in range(num_options):
            pi_o = policies[o]
            for s in range(S):
                q_val = 0.0
                for a in range(A):
                    if pi_o[s, a] > 0.0:
                        # Expected immediate reward + discounted expected upon-arrival value
                        expected_future_u = np.dot(P[s, a], U[o])
                        q_val += pi_o[s, a] * (R[s, a] + gamma * expected_future_u)
                Q_next[s, o] = q_val
        Q = Q_next

    # Re-sync V and U one final time to ensure they completely match the final updated Q
    V = np.max(Q, axis=1)
    for o in range(num_options):
        beta = terminations[o]
        U[o] = (1.0 - beta) * Q[:, o] + beta * V

    return {
        'Q_options': np.round(Q, 4).tolist(),
        'V': np.round(V, 4).tolist(),
        'U': np.round(U, 4).tolist()
    }