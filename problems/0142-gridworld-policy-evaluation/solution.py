def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    # Initialize V(s) = 0 for all states
    V = [[0.0 for _ in range(5)] for _ in range(5)]
    
    # Define terminal states (corners)
    terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}
    
    # Define action effects (row, col) changes
    actions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    # Iterative policy evaluation
    while True:
        delta = 0.0
        
        # Create a copy of V to update simultaneously
        V_new = [row[:] for row in V]
        
        # Iterate over all states
        for i in range(5):
            for j in range(5):
                # Skip terminal states (value fixed at 0)
                if (i, j) in terminal_states:
                    V_new[i][j] = 0.0
                    continue
                
                # Compute Bellman expectation for this state
                state_value = 0.0
                
                # Get action probabilities for this state
                action_probs = policy.get((i, j), {})
                
                # Sum over all actions
                for action, prob in action_probs.items():
                    # Get the next state
                    di, dj = actions[action]
                    next_i = i + di
                    next_j = j + dj
                    
                    # Check boundaries - if action would move off grid, stay in same state
                    if not (0 <= next_i < 5 and 0 <= next_j < 5):
                        next_i, next_j = i, j
                    
                    # Bellman expectation equation:
                    # V(s) = sum_a π(a|s) * [R + γ * V(s')]
                    reward = -1.0  # Constant reward for each move
                    state_value += prob * (reward + gamma * V[next_i][next_j])
                
                V_new[i][j] = state_value
                
                # Track the maximum change
                delta = max(delta, abs(V_new[i][j] - V[i][j]))
        
        # Update V
        V = V_new
        
        # Check convergence
        if delta < threshold:
            break
    
    return V