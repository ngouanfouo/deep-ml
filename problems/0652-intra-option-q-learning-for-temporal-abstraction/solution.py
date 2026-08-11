import numpy as np

def intra_option_q_learning(
    num_states: int,
    num_actions: int,
    num_options: int,
    option_policies: list,
    option_terminations: list,
    initiation_sets: list,
    transitions: list,
    alpha: float,
    gamma: float
) -> list:
    """
    Learn option-value function Q(s, o) using intra-option Q-learning.

    Args:
        num_states: Total number of states
        num_actions: Total number of primitive actions
        num_options: Total number of options
        option_policies: option_policies[o][s][a] = P(a | s, o)
        option_terminations: option_terminations[o][s] = P(terminate | s, o)
        initiation_sets: initiation_sets[o] = list of states where option o can start
        transitions: List of (state, action, reward, next_state) tuples
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        2D list of Q-values, shape (num_states, num_options), rounded to 4 decimals.
    """
    # Initialize Q-values to zero
    Q = np.zeros((num_states, num_options), dtype=float)
    
    # Convert to numpy arrays for easier indexing
    policies = np.array(option_policies)
    terminations = np.array(option_terminations)
    
    def get_best_option_value(state):
        """Return max Q-value over options that can initiate in this state."""
        max_value = float('-inf')
        for o in range(num_options):
            if state in initiation_sets[o]:
                max_value = max(max_value, Q[state, o])
        # If no option can initiate, return 0
        return max_value if max_value != float('-inf') else 0.0
    
    # Process each transition
    for s, a, r, next_s in transitions:
        # For each option, check if it can take action a in state s
        for o in range(num_options):
            # Skip if option cannot initiate in state s
            if s not in initiation_sets[o]:
                continue
            
            # Check if option's policy assigns positive probability to action a
            if policies[o, s, a] > 0:
                # Compute termination probability in next state
                beta = terminations[o, next_s]
                
                # Compute continuation value U(next_s, o)
                # U(s', o) = (1 - beta) * Q(s', o) + beta * max_{o' in I(s')} Q(s', o')
                if next_s in initiation_sets[o]:
                    # Option can continue in next state if it's in the initiation set
                    # Even if beta=1, the option terminates, so the value is the max over options
                    best_next = get_best_option_value(next_s)
                    U_next = (1 - beta) * Q[next_s, o] + beta * best_next
                else:
                    # If option cannot initiate in next state, it must terminate
                    # (beta should be 1 in this case, but we handle it generally)
                    best_next = get_best_option_value(next_s)
                    U_next = best_next
                
                # Compute TD error
                delta = r + gamma * U_next - Q[s, o]
                
                # Update Q-value
                Q[s, o] += alpha * delta
    
    # Round to 4 decimal places
    Q_rounded = np.round(Q, 4)
    
    # Convert to list of lists
    return Q_rounded.tolist()