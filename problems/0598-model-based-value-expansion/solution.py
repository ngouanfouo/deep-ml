import numpy as np

def model_based_value_expansion(Q, transitions, rewards_model, V, experiences, H, gamma, alpha, terminal_states):
    """
    Perform model-based value expansion updates on Q-values.
    
    Args:
        Q: np.ndarray of shape (num_states, num_actions), initial Q-values
        transitions: dict mapping (state, action) -> next_state
        rewards_model: dict mapping (state, action) -> reward
        V: np.ndarray of shape (num_states,), value function for bootstrapping
        experiences: list of (state, action) tuples to update
        H: int, model rollout horizon
        gamma: float, discount factor
        alpha: float, learning rate
        terminal_states: list of int, terminal state indices
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), updated Q-values rounded to 4 decimals
    """
    
    # Make a copy of Q to avoid modifying the original
    Q_updated = Q.copy()
    
    # Process each experience sequentially
    for state, action in experiences:
        # Current discount factor for accumulating rewards
        discount = 1.0
        
        # Accumulated return G
        G = 0.0
        
        # Current state in the rollout
        current_state = state
        current_action = action
        
        # Simulate H steps using the learned model
        for step in range(H):
            # Check if current state is terminal
            if current_state in terminal_states:
                # If terminal, stop rollout and don't bootstrap
                break
            
            # Get reward and next state from learned models
            reward = rewards_model.get((current_state, current_action), 0.0)
            next_state = transitions.get((current_state, current_action), current_state)
            
            # Accumulate discounted reward
            G += discount * reward
            
            # Update discount factor for next step
            discount *= gamma
            
            # Move to next state
            current_state = next_state
            
            # If we reached the horizon or terminal state, stop
            if step == H - 1:
                break
            
            # Select greedy action at the next state
            # argmax over Q-values for the current state
            current_action = np.argmax(Q_updated[current_state])
        
        # After rollout, bootstrap with value function if not terminal
        if current_state not in terminal_states:
            G += discount * V[current_state]
        
        # Update Q-value using TD-style update
        current_q = Q_updated[state, action]
        Q_updated[state, action] = current_q + alpha * (G - current_q)
    
    # Round to 4 decimal places
    return np.round(Q_updated, 4)