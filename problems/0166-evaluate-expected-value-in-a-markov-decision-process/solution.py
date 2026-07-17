import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    # Your code here
    expected_value = 0.0
    
    # Get the transition probabilities for this state and action
    transitions = P[state][action]
    
    # Get the rewards for this state and action
    rewards = R[state][action]
    
    # Iterate over all possible next states
    for next_state, prob in transitions.items():
        # Get the reward for this transition
        reward = rewards.get(next_state, 0.0)
        
        # Get the value of the next state
        # Handle both integer and string state indices
        if isinstance(next_state, int):
            next_value = V[next_state]
        else:
            # If state is a string, we need to map it to an index
            # Assuming states are in order: 0, 1, 2, ...
            # This is a simplification - in practice you might need a state_to_idx mapping
            next_value = V[int(next_state)] if str(next_state).isdigit() else 0.0
        
        # Bellman expectation equation:
        # Q(s,a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
        expected_value += prob * (reward + gamma * next_value)
    
    return expected_value