import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    n_states = len(V)
    new_V = np.zeros(n_states)
    
    for s in range(n_states):
        # Initialize best action value to negative infinity
        best_action_value = -np.inf
        
        # Iterate over all actions for this state
        for action in transitions[s]:
            action_value = 0.0
            
            # Iterate over all possible outcomes of this action
            for prob, next_state, reward, done in transitions[s][action]:
                if done:
                    # If episode ends, no future value
                    action_value += prob * reward
                else:
                    # Bellman equation: sum of (prob * (reward + gamma * V[next_state]))
                    action_value += prob * (reward + gamma * V[next_state])
            
            # Keep the maximum action value
            if action_value > best_action_value:
                best_action_value = action_value
        
        # Update value for this state
        new_V[s] = best_action_value
    
    return new_V