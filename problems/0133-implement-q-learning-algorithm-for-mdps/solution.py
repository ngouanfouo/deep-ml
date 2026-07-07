import numpy as np

def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    """
    Implements tabular Q-Learning.
    
    Args:
        num_states: int, total number of states
        num_actions: int, total number of actions
        P: 3D NumPy array of shape (num_states, num_actions, num_states)
        R: 2D NumPy array of shape (num_states, num_actions)
        terminal_states: list or 1D NumPy array of terminal state indices
        alpha: float, learning rate
        gamma: float, discount factor
        epsilon: float, exploration rate for epsilon-greedy selection
        num_episodes: int, number of simulation episodes
        
    Returns:
        Q: 2D NumPy array of shape (num_states, num_actions) representing the learned Q-table
    """
    # 1. Initialize the Q-table to zeros
    Q = np.zeros((num_states, num_actions), dtype=float)
    
    # Identify non-terminal states to start episodes from safely
    terminal_set = set(terminal_states)
    non_terminal_states = [s for s in range(num_states) if s not in terminal_set]
    
    if not non_terminal_states:
        return Q

    # 2. Main loop over episodes
    for episode in range(num_episodes):
        # Sample a random starting non-terminal state
        state = np.random.choice(non_terminal_states)
        
        # Loop until a terminal state is reached
        while state not in terminal_set:
            # 3. Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                # Explore: pick a random action uniformly
                action = np.random.choice(num_actions)
            else:
                # Exploit: standard argmax (deterministic tie-breaking to match expected seed states)
                action = np.argmax(Q[state])
            
            # 4. Get the reward for this state-action pair
            reward = R[state, action]
            
            # 5. Sample the next state based on the transition probabilities matrix P
            transition_probs = P[state, action]
            next_state = np.random.choice(num_states, p=transition_probs)
            
            # 6. Apply the Temporal Difference (TD) Q-Learning Update Rule
            if next_state in terminal_set:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
                
            Q[state, action] += alpha * (target - Q[state, action])
            
            # Update current state tracker
            state = next_state
            
    return Q