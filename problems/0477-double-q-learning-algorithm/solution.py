import numpy as np

def double_q_learning(
    num_states: int,
    num_actions: int,
    experiences: list,
    alpha: float,
    gamma: float,
    coin_flips: list
) -> tuple:
    """
    Perform Double Q-learning updates on a sequence of experiences.
    
    Args:
        num_states: Number of states in the environment
        num_actions: Number of possible actions
        experiences: List of (state, action, reward, next_state, done) tuples
        alpha: Learning rate
        gamma: Discount factor
        coin_flips: List of 0 or 1 indicating which Q-table to update
        
    Returns:
        Tuple of (Q1, Q2) numpy arrays, each of shape (num_states, num_actions)
    """
    # Initialize two Q-tables to zeros
    Q1 = np.zeros((num_states, num_actions), dtype=float)
    Q2 = np.zeros((num_states, num_actions), dtype=float)

    for (state, action, reward, next_state, done), coin in zip(experiences, coin_flips):
        if coin == 0:
            # Update Q1, evaluate with Q2
            if done:
                target = reward
            else:
                # Action selection uses Q1; action evaluation uses Q2
                best_action = int(np.argmax(Q1[next_state]))
                target = reward + gamma * Q2[next_state, best_action]

            Q1[state, action] += alpha * (target - Q1[state, action])

        else:
            # Update Q2, evaluate with Q1
            if done:
                target = reward
            else:
                # Action selection uses Q2; action evaluation uses Q1
                best_action = int(np.argmax(Q2[next_state]))
                target = reward + gamma * Q1[next_state, best_action]

            Q2[state, action] += alpha * (target - Q2[state, action])

    return Q1, Q2