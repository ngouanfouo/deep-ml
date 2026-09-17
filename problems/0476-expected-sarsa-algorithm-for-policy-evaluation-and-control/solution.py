import numpy as np

def expected_sarsa(Q: np.ndarray, episode: list, alpha: float, gamma: float, epsilon: float) -> np.ndarray:
    """
    Perform Expected SARSA updates on a Q-table given an episode of experience.
    
    Args:
        Q: Q-table of shape (num_states, num_actions)
        episode: list of (state, action, reward, next_state, done) tuples
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate for epsilon-greedy policy
    
    Returns:
        Updated Q-table (numpy array)
    """
    # Work on a copy so the original is not modified
    Q = np.array(Q, dtype=float, copy=True)
    num_actions = Q.shape[1]

    for state, action, reward, next_state, done in episode:
        if done:
            expected_future = 0.0
        else:
            # Epsilon-greedy action probabilities at next_state, based on current Q
            q_next = Q[next_state]
            greedy_action = int(np.argmax(q_next))

            probs = np.full(num_actions, epsilon / num_actions, dtype=float)
            probs[greedy_action] += (1.0 - epsilon)

            expected_future = float(np.sum(probs * q_next))

        td_target = reward + gamma * expected_future
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

    return Q