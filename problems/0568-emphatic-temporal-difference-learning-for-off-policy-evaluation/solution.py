def emphatic_td(trajectory: list, gamma: float, lam: float, alpha: float, target_policy: dict, behavior_policy: dict, interest: dict, initial_values: dict) -> dict:
    """
    Emphatic TD(lambda) for off-policy tabular policy evaluation.
    
    Args:
        trajectory: list of (state, action, reward, next_state) tuples
        gamma: discount factor
        lam: lambda for eligibility traces
        alpha: step size
        target_policy: dict (state, action) -> probability under target policy
        behavior_policy: dict (state, action) -> probability under behavior policy
        interest: dict state -> interest value (default 0 for missing states)
        initial_values: dict state -> initial value estimate
    
    Returns:
        Dict mapping state -> rounded value estimate
    """
    # Initialize values
    V = {state: float(val) for state, val in initial_values.items()}
    
    # Initialize traces
    e = {state: 0.0 for state in initial_values.keys()}
    F = 0.0
    rho_prev = 0.0
    
    for state, action, reward, next_state in trajectory:
        # Get interest for current state
        I = interest.get(state, 0.0)
        
        # Calculate importance sampling ratio
        pi_prob = target_policy.get((state, action), 0.0)
        b_prob = behavior_policy.get((state, action), 0.0)
        rho = pi_prob / b_prob if b_prob > 0.0 else 0.0
        
        # Update follow-on trace and emphasis
        F = gamma * rho_prev * F + I
        M = lam * I + (1.0 - lam) * F
        
        # Ensure current state is tracked in traces and values if not present
        if state not in e:
            e[state] = 0.0
        if state not in V:
            V[state] = 0.0
            
        # Calculate TD error
        v_next = 0.0 if next_state is None else V.get(next_state, 0.0)
        delta = reward + gamma * v_next - V[state]
        
        # Update eligibility traces for all tracked states
        for s in e.keys():
            if s == state:
                e[s] = rho * (gamma * lam * e[s] + M)
            else:
                e[s] = rho * gamma * lam * e[s]
                
        # Update value estimates using the eligibility traces
        for s in V.keys():
            V[s] += alpha * delta * e.get(s, 0.0)
            
        # Update previous rho for the next transition
        rho_prev = rho

    # Return only the initial states, sorted and rounded to 4 decimal places
    return {state: round(V[state], 4) for state in sorted(initial_values.keys())}