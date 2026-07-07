import numpy as np

def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    """
    Simulates a Markov Chain given a transition matrix, initial state, and step count.
    
    Args:
        transition_matrix: 2D NumPy array of shape (num_states, num_states)
                           where transition_matrix[i, j] is the probability of P(X_{t+1}=j | X_t=i)
        initial_state: Integer index representing the starting state
        num_steps: Number of transitions to simulate
        
    Returns:
        history: 1D NumPy array of shape (num_steps + 1,) containing the sequence of state indices
    """
    # 1. Initialize the history array to hold the initial state plus all future steps
    history = np.zeros(num_steps + 1, dtype=int)
    history[0] = initial_state
    
    current_state = initial_state
    num_states = transition_matrix.shape[1]
    state_indices = np.arange(num_states)
    
    # 2. Iteratively sample the next state
    for t in range(1, num_steps + 1):
        # Retrieve the transition probability distribution row for the current state
        probabilities = transition_matrix[current_state]
        
        # Select the next state using the probability distribution
        next_state = np.random.choice(state_indices, p=probabilities)
        
        # Log the selected state and update the tracker
        history[t] = next_state
        current_state = next_state
        
    return history