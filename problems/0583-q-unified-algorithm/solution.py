import numpy as np

def q_sigma_return(
    states: list,
    actions: list,
    rewards: list,
    sigmas: list,
    Q: np.ndarray,
    pi: np.ndarray,
    b: np.ndarray,
    gamma: float
) -> float:
    """
    Compute the n-step Q(sigma) return.
    
    Args:
        states: List of n+1 state indices [S_t, S_{t+1}, ..., S_{t+n}]
        actions: List of n+1 action indices [A_t, A_{t+1}, ..., A_{t+n}]
        rewards: List of n rewards [R_{t+1}, R_{t+2}, ..., R_{t+n}]
        sigmas: List of n+1 sigma values [sigma_t, sigma_{t+1}, ..., sigma_{t+n}]
        Q: Q-value table, shape (n_states, n_actions)
        pi: Target policy probabilities, shape (n_states, n_actions)
        b: Behavior policy probabilities, shape (n_states, n_actions)
        gamma: Discount factor
    
    Returns:
        The n-step Q(sigma) return as a float.
    """
    n = len(rewards)
    
    # Bootstrap value at the horizon terminal state-action pair (S_{t+n}, A_{t+n})
    G = float(Q[states[n], actions[n]])
    
    # Work backward along the trajectory sequence from k = t+n-1 down to t
    for k in range(n - 1, -1, -1):
        s_next = states[k + 1]
        a_next = actions[k + 1]
        r_next = rewards[k]
        sigma_next = sigmas[k + 1]
        
        # Calculate policy elements at the next state-action node
        pi_next = pi[s_next, a_next]
        b_next = b[s_next, a_next]
        rho_next = pi_next / b_next if b_next > 0 else 0.0
        
        # Expected value under the target policy at the next state
        V_bar_next = np.sum(pi[s_next] * Q[s_next])
        
        # Interpolate between Sarsa-style sampling and Tree Backup expectations
        trace_weight = sigma_next * rho_next + (1.0 - sigma_next) * pi_next
        
        # Apply the recursive update equation
        G = r_next + gamma * trace_weight * (G - Q[s_next, a_next]) + gamma * V_bar_next
        
    return G