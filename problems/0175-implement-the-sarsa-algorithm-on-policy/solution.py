def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    """
    Perform SARSA updates on the given environment transitions.
    Args:
        transitions (dict): mapping (state, action) -> (reward, next_state)
        initial_states (list): list of starting states to simulate episodes from
        alpha (float): learning rate
        gamma (float): discount factor
        max_steps (int): maximum steps allowed per episode
    Returns:
        dict: final Q-table as a dictionary {(state, action): value}
    """
    # Initialize all Q-values present in transitions to 0.0
    Q = {k: 0.0 for k in transitions.keys()}
    
    # Helper to group available actions by state
    state_actions = {}
    for state, action in transitions.keys():
        if state not in state_actions:
            state_actions[state] = []
        state_actions[state].append(action)
    
    # Ensure actions are sorted alphabetically for deterministic tie-breaking
    for state in state_actions:
        state_actions[state].sort()

    def get_greedy_action(state):
        if state == 'terminal' or state not in state_actions:
            return None
        
        # Greedy selection: choose action that maximizes Q(state, action)
        # Ties are broken by alphabetical order due to the sorting above
        best_action = None
        best_val = -float('inf')
        for action in state_actions[state]:
            val = Q[(state, action)]
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

    # Simulate an episode for each starting state sequentially
    for start_state in initial_states:
        state = start_state
        action = get_greedy_action(state)
        steps = 0
        
        while state != 'terminal' and steps < max_steps and action is not None:
            # Look up deterministic environment transition outcome
            reward, next_state = transitions[(state, action)]
            
            # Select the next action greedily from the next state (SARSA)
            next_action = get_greedy_action(next_state)
            
            # Compute TD target future value estimate
            if next_state == 'terminal' or next_action is None:
                next_q = 0.0
            else:
                next_q = Q[(next_state, next_action)]
                
            # Perform sequential Q-value temporal difference update
            Q[(state, action)] += alpha * (reward + gamma * next_q - Q[(state, action)])
            
            # Transition forward
            state = next_state
            action = next_action
            steps += 1
            
    return Q