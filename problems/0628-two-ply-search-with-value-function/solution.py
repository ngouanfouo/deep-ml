def two_ply_search(state: int, transitions: dict, value_function: dict, gamma: float) -> dict:
    """
    Perform a two-ply lookahead search to compute action values.
    
    Args:
        state: Current state (integer)
        transitions: Dict mapping (state, action) -> list of (prob, next_state, reward)
        value_function: Dict mapping state -> estimated value (float)
        gamma: Discount factor
        
    Returns:
        Dict mapping each available action to its two-ply Q-value (rounded to 4 decimals)
    """
    # Find all actions available from the current state
    actions = [a for (s, a) in transitions.keys() if s == state]
    
    result = {}
    
    for action in actions:
        q_value = 0.0
        
        for prob, next_state, reward in transitions[(state, action)]:
            # Check whether the next state has any available actions
            next_actions = [b for (s, b) in transitions.keys() if s == next_state]
            
            if next_actions:
                # Two-ply: choose the best second-step action greedily
                best_q2 = float('-inf')
                for next_action in next_actions:
                    q2 = 0.0
                    for p2, s2, r2 in transitions[(next_state, next_action)]:
                        q2 += p2 * (r2 + gamma * value_function.get(s2, 0.0))
                    if q2 > best_q2:
                        best_q2 = q2
                q_value += prob * (reward + gamma * best_q2)
            else:
                # Fall back to one-ply evaluation using the value function directly
                q_value += prob * (reward + gamma * value_function.get(next_state, 0.0))
        
        result[action] = round(q_value, 4)
    
    return result