import numpy as np

def horde_parallel_gvfs(
    experience: list,
    demons: list,
    behavior_policy: np.ndarray,
    n_states: int,
    n_actions: int,
    alpha: float
) -> list:
    """
    Learn multiple General Value Functions in parallel from a single
    experience stream using off-policy TD(lambda) with importance sampling.
    
    Args:
        experience: List of (state, action, next_state) tuples.
        demons: List of demon configs, each a dict with keys:
            'cumulant': np.ndarray of shape (n_states, n_actions, n_states)
            'gamma': float, discount factor
            'target_policy': np.ndarray of shape (n_states, n_actions)
            'lambd': float, eligibility trace decay
        behavior_policy: np.ndarray of shape (n_states, n_actions)
        n_states: Number of states.
        n_actions: Number of actions.
        alpha: Learning rate.
    
    Returns:
        List of lists, one per demon, each of length n_states,
        with values rounded to 4 decimal places.
    """
    # Initialize separate state-value tables and trace trackers for each demon
    num_demons = len(demons)
    V_demons = [np.zeros(n_states, dtype=float) for _ in range(num_demons)]
    e_demons = [np.zeros(n_states, dtype=float) for _ in range(num_demons)]
    
    # Process each transition (s, a, s') from the joint stream of experience
    for s, a, s_next in experience:
        b_prob = behavior_policy[s, a]
        
        for i, demon in enumerate(demons):
            cumulant_matrix = demon['cumulant']
            gamma = demon['gamma']
            pi = demon['target_policy']
            lambd = demon['lambd']
            
            # 1. Fetch the unique pseudo-reward/cumulant signal for this demon
            c = cumulant_matrix[s, a, s_next]
            
            # 2. Compute the off-policy importance sampling correction ratio
            if b_prob > 0.0:
                rho = pi[s, a] / b_prob
            else:
                rho = 0.0
                
            # 3. Compute the temporal difference prediction error
            td_error = c + gamma * V_demons[i][s_next] - V_demons[i][s]
            
            # 4. Update the eligibility trace according to the required sequential order:
            # First decay the trace vector, add current state flag, then rescale by rho
            e_demons[i] = gamma * lambd * e_demons[i]
            e_demons[i][s] += 1.0
            e_demons[i] = rho * e_demons[i]
            
            # 5. Broadcast the value function update using the credit assigned across the traces
            V_demons[i] += alpha * td_error * e_demons[i]
            
    # Format and round results to 4 decimal places as requested
    output = []
    for V in V_demons:
        output.append([round(float(v), 4) for v in V])
        
    return output