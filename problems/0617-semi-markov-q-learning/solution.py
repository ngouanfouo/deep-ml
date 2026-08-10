import numpy as np

def semi_markov_q_learning(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float
) -> list:
    """
    Implement Semi-Markov Q-Learning with variable action durations.
    
    Args:
        episodes: List of episodes, each a list of
                  (state, action, reward, next_state, duration, done) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Per-unit-time discount factor
        alpha: Learning rate
    
    Returns:
        Q-value table as nested list of shape (n_states, n_actions),
        rounded to 4 decimal places
    """
    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))
    
    # Process each episode
    for episode in episodes:
        # Process each transition in the episode
        for transition in episode:
            state, action, reward, next_state, duration, done = transition
            
            # Compute the target for SMDP Q-learning
            # If done (terminal state), the target is just the reward
            if done:
                target = reward
            else:
                # For non-terminal states:
                # target = reward + gamma^duration * max_a' Q(next_state, a')
                # For standard Q-learning with duration=1, this reduces to:
                # target = reward + gamma * max_a' Q(next_state, a')
                max_next_q = np.max(Q[next_state])
                target = reward + (gamma ** duration) * max_next_q
            
            # Q-learning update: Q(s,a) += alpha * (target - Q(s,a))
            Q[state, action] += alpha * (target - Q[state, action])
    
    # Convert to nested list and round to 4 decimal places
    return np.round(Q, 4).tolist()