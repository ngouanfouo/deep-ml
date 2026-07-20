import numpy as np

def sarsa_lambda(n_states: int, n_actions: int, transitions: dict, terminal_states: list, gamma: float, alpha: float, lam: float, epsilon: float, num_episodes: int, seed: int = 42) -> list:
    """
    Sarsa(lambda) with accumulating eligibility traces.
    
    Args:
        n_states: Number of states
        n_actions: Number of actions
        transitions: Dict mapping (state, action) -> (next_state, reward)
        terminal_states: List of terminal state indices
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda for eligibility trace decay
        epsilon: Exploration parameter
        num_episodes: Number of training episodes
        seed: Random seed
    
    Returns:
        Q-table as 2D list rounded to 4 decimal places
    """
    # Set up random state generator for reproducibility
    rng = np.random.RandomState(seed)
    
    # Initialize action-value table to zeros
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    
    # Convert terminal states to a set for O(1) membership checks
    terminals = set(terminal_states)
    
    # Identify non-terminal states available for choosing starting points
    non_terminal_states = [s for s in range(n_states) if s not in terminals]
    
    if not non_terminal_states:
        return [[0.0] * n_actions for _ in range(n_states)]

    def select_action(state, q_table, eps):
        # Epsilon-greedy selection
        if rng.uniform(0, 1) < eps:
            return rng.randint(0, n_actions)
        else:
            # Greedy action selection breaking ties by selecting the smallest index
            state_qs = q_table[state]
            max_q = np.max(state_qs)
            return int(np.min(np.where(state_qs == max_q)[0]))

    for _ in range(num_episodes):
        # Reset accumulating eligibility traces to zero at the start of each episode
        E = np.zeros((n_states, n_actions), dtype=np.float64)
        
        # Select initial non-terminal state uniformly at random
        state = rng.choice(non_terminal_states)
        action = select_action(state, Q, epsilon)
        
        while state not in terminals:
            # Query the deterministic environment dynamics
            next_state, reward = transitions[(state, action)]
            
            # Determine target value depending on whether the next state is terminal
            if next_state in terminals:
                next_action = None
                q_next = 0.0
            else:
                next_action = select_action(next_state, Q, epsilon)
                q_next = Q[next_state, next_action]
                
            # Compute the TD error delta
            delta = reward + gamma * q_next - Q[state, action]
            
            # Increment the trace for the current state-action pair (accumulating trace)
            E[state, action] += 1.0
            
            # Apply TD error update globally across all eligible table entries
            Q += alpha * delta * E
            
            # Decay the eligibility traces
            E *= gamma * lam
            
            # Transition to the next time step variables
            state = next_state
            if next_action is not None:
                action = next_action

    # Convert table values to a 2D list rounded to 4 decimal places
    return [[round(float(val), 4) for val in row] for row in Q]