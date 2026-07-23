import numpy as np

def differential_sarsa(transitions: dict, initial_state: str, alpha: float, beta: float, num_steps: int) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    
    Args:
        transitions: dict mapping (state, action) -> (reward, next_state)
        initial_state: starting state
        alpha: step size for Q-value updates
        beta: step size for average reward estimate
        num_steps: number of steps to simulate
    
    Returns:
        Tuple of (Q, R_bar) where Q is a dict {(state, action): float}
        and R_bar is a float.
    """
    # Initialize Q-values to zero for all state-action pairs in transitions
    Q = {key: 0.0 for key in transitions.keys()}
    
    # Initialize average reward estimate
    R_bar = 0.0
    
    # Get all actions available from each state for greedy selection
    # We need to know which actions are possible from each state
    state_actions = {}
    for (state, action) in transitions.keys():
        if state not in state_actions:
            state_actions[state] = []
        state_actions[state].append(action)
    
    # Start from the initial state
    current_state = initial_state
    
    # Select the first action greedily (ties broken by lexicographically smallest)
    # Get all actions available from current_state
    available_actions = state_actions.get(current_state, [])
    # Find the action with max Q-value (ties broken by lexicographically smallest)
    current_action = min(available_actions, key=lambda a: (-Q[(current_state, a)], a))
    
    # Process each step
    for _ in range(num_steps):
        # Get reward and next state for current transition
        reward, next_state = transitions[(current_state, current_action)]
        
        # Select next action greedily from next_state
        next_available_actions = state_actions.get(next_state, [])
        next_action = min(next_available_actions, key=lambda a: (-Q[(next_state, a)], a))
        
        # Compute differential TD error
        # delta = reward - R_bar + Q(next_state, next_action) - Q(current_state, current_action)
        delta = reward - R_bar + Q[(next_state, next_action)] - Q[(current_state, current_action)]
        
        # Update average reward estimate
        R_bar += beta * delta
        
        # Update Q-value
        Q[(current_state, current_action)] += alpha * delta
        
        # Move to next state-action pair
        current_state = next_state
        current_action = next_action
    
    # Round Q-values to 4 decimal places
    Q_rounded = {key: round(float(val), 4) for key, val in Q.items()}
    
    # Round R_bar to 4 decimal places
    R_bar_rounded = round(float(R_bar), 4)
    
    return Q_rounded, R_bar_rounded