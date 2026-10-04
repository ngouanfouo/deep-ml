import numpy as np

def forward_backward_td(episodes: list, n_states: int, gamma: float, lam: float, alpha: float) -> dict:
    """
    Compute value function estimates using both the forward view (lambda-returns)
    and backward view (eligibility traces) of TD(lambda) in offline mode.
    
    Args:
        episodes: List of episodes, each a list of (state, reward) tuples.
        n_states: Number of states in the environment.
        gamma: Discount factor.
        lam: Lambda parameter for trace decay.
        alpha: Learning rate.
    
    Returns:
        Dictionary with 'forward' and 'backward' keys, each a list of floats
        rounded to 4 decimal places.
    """
    # ---------------- Forward View ----------------
    V_forward = np.zeros(n_states)
    
    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue
        
        V_start = V_forward.copy()
        G_lambda = np.zeros(T)
        
        # Compute lambda-returns backwards
        for t in range(T - 1, -1, -1):
            r = episode[t][1]
            if t == T - 1:
                G_lambda[t] = r
            else:
                s_next = episode[t + 1][0]
                G_lambda[t] = r + gamma * ((1 - lam) * V_start[s_next] + lam * G_lambda[t + 1])
        
        # Accumulate batch updates
        updates = np.zeros(n_states)
        for t in range(T):
            s = episode[t][0]
            updates[s] += G_lambda[t] - V_start[s]
        
        V_forward += alpha * updates
    
    # ---------------- Backward View ----------------
    V_backward = np.zeros(n_states)
    
    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue
        
        V_start = V_backward.copy()
        e = np.zeros(n_states)          # eligibility traces
        accum = np.zeros(n_states)      # accumulated weight updates
        
        for t in range(T):
            s = episode[t][0]
            r = episode[t][1]
            
            if t == T - 1:
                V_next = 0.0
            else:
                s_next = episode[t + 1][0]
                V_next = V_start[s_next]
            
            delta = r + gamma * V_next - V_start[s]
            
            # Decay and update eligibility traces
            e *= gamma * lam
            e[s] += 1.0
            
            # Accumulate updates
            accum += alpha * delta * e
        
        V_backward += accum
    
    # ---------------- Format Output ----------------
    return {
        'forward': [round(float(v), 4) for v in V_forward],
        'backward': [round(float(v), 4) for v in V_backward]
    }