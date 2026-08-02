import numpy as np

def options_value_iteration(n_states, terminal_states, transitions, options, gamma, theta=1e-10):
    """
    Perform SMDP value iteration using temporally extended options.
    
    Args:
        n_states: int, number of states
        terminal_states: set of terminal state indices
        transitions: dict, transitions[s][a] = [(next_state, prob, reward), ...]
        options: list of option dicts with 'initiation', 'policy', 'termination'
        gamma: float, discount factor
        theta: float, convergence threshold
    
    Returns:
        V: list of floats (rounded to 4 decimals), optimal value per state
        policy: list of ints, optimal option index per state
    """
    num_options = len(options)
    
    # Storage maps for the models of each option: R_model[o][s] and F_model[o][s, s_next]
    R_model = {}
    F_model = {}
    
    # Phase 1: Compute Option Models via solving linear equation systems
    for o_idx, option in enumerate(options):
        init_set = option['initiation']
        policy = option['policy']
        beta = option['termination']
        
        # We solve systems of equations only across the states inside the initiation set.
        # However, to easily handle transitions out of the initiation set (or to terminal states),
        # we can formulate a matrix equation for all non-terminal states.
        # System: (I - A) X = B
        
        # 1. Multi-step Reward Model R(s, o)
        A_R = np.zeros((n_states, n_states))
        B_R = np.zeros(n_states)
        
        # 2. Transition Multi-step Model F(s' | s, o)
        A_F = np.zeros((n_states, n_states))
        B_F = np.zeros((n_states, n_states))
        
        for s in range(n_states):
            if s in terminal_states or s not in init_set:
                # Terminal states or out-of-initiation states terminate or are invalid.
                # Setting coefficient to identity yields a solution value of 0.
                A_R[s, s] = 1.0
                A_F[s, s] = 1.0
                continue
                
            a = policy[s]
            trans_list = transitions.get(s, {}).get(a, [])
            
            A_R[s, s] = 1.0
            A_F[s, s] = 1.0
            
            for next_state, prob, reward in trans_list:
                beta_next = beta.get(next_state, 0.0)
                
                # Update reward system rules
                B_R[s] += prob * reward
                A_R[s, next_state] -= prob * gamma * (1.0 - beta_next)
                
                # Update transition probability system rules
                B_F[s, next_state] += prob * gamma * beta_next
                A_F[s, next_state] -= prob * gamma * (1.0 - beta_next)
                
        # Solve the linear systems
        R_sol = np.linalg.solve(A_R, B_R)
        F_sol = np.linalg.solve(A_F, B_F)
        
        R_model[o_idx] = R_sol
        F_model[o_idx] = F_sol

    # Phase 2: Perform SMDP Value Iteration
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        V_old = V.copy()
        
        for s in range(n_states):
            if s in terminal_states:
                V[s] = 0.0
                continue
                
            best_val = -float('inf')
            
            # Find the best option available at current state
            for o_idx, option in enumerate(options):
                if s in option['initiation']:
                    # Expected option value: R(s,o) + sum_{s'} F(s'|s,o) * V(s')
                    val = R_model[o_idx][s] + np.dot(F_model[o_idx][s], V_old)
                    if val > best_val:
                        best_val = val
                        
            if best_val != -float('inf'):
                V[s] = best_val
                delta = max(delta, abs(V[s] - V_old[s]))
                
        if delta < theta:
            break

    # Extract the optimal policy indices matching the computed value function
    policy = [0] * n_states
    for s in range(n_states):
        if s in terminal_states:
            policy[s] = 0
            continue
            
        best_val = -float('inf')
        best_opt = 0
        
        for o_idx, option in enumerate(options):
            if s in option['initiation']:
                val = R_model[o_idx][s] + np.dot(F_model[o_idx][s], V)
                if val > best_val:
                    best_val = val
                    best_opt = o_idx
                    
        policy[s] = best_opt
        
    V_rounded = [round(float(v), 4) for v in V]
    
    return V_rounded, policy