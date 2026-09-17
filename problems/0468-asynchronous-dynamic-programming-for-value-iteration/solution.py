import numpy as np

def async_value_iteration(num_states: int, transitions: list, gamma: float, update_order: list) -> list:
    """
    Perform asynchronous value iteration on an MDP.
    
    Args:
        num_states: Number of states in the MDP
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        update_order: Sequence of state indices to update (in order)
    
    Returns:
        List of float values representing the value function
    """
    # Initialize value function to zeros
    V = [0.0] * num_states

    for s in update_order:
        # Skip states with no available actions
        if s < 0 or s >= num_states:
            continue
        actions = transitions[s] if s < len(transitions) else []
        if not actions:
            continue

        # Bellman optimality backup: max over actions of expected return
        best_value = float('-inf')
        for action in actions:
            if not action:
                # An action with no transitions contributes nothing useful;
                # treat its expected return as 0.
                value = 0.0
            else:
                value = 0.0
                for prob, next_state, reward in action:
                    value += prob * (reward + gamma * V[next_state])
            if value > best_value:
                best_value = value

        if best_value > float('-inf'):
            V[s] = best_value

    return V