import numpy as np

def on_policy_sample(P, R, policy, start_state, terminal_states, gamma,
                     num_episodes, max_steps, seed=42):
    """
    Sample trajectories on-policy and compute visitation statistics.
    """
    np.random.seed(seed)
    terminal_set = set(terminal_states)

    n_states = P.shape[0]
    n_actions = P.shape[1]

    # Counters
    state_visit_counts = np.zeros(n_states, dtype=int)
    state_action_counts = np.zeros((n_states, n_actions), dtype=int)
    episode_returns = []

    for _ in range(num_episodes):
        state = start_state
        total_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            # If we're in a terminal state, the episode ends (no action taken)
            if state in terminal_set:
                break

            # Count the visit to this (non-terminal) state before taking an action
            state_visit_counts[state] += 1

            # Sample action from the policy
            action = int(np.random.choice(n_actions, p=policy[state]))
            state_action_counts[state, action] += 1

            # Sample next state from transition distribution
            next_state = int(np.random.choice(n_states, p=P[state, action]))

            # Accumulate discounted reward
            reward = R[state, action, next_state]
            total_return += discount * reward
            discount *= gamma

            state = next_state

        episode_returns.append(float(total_return))

    # State visitation frequency: fraction of total visits per state
    total_visits = int(state_visit_counts.sum())
    if total_visits == 0:
        state_visitation_freq = [0.0] * n_states
    else:
        state_visitation_freq = [
            round(float(c) / total_visits, 4) for c in state_visit_counts
        ]

    # State-action counts as nested lists of ints
    state_action_counts_out = state_action_counts.tolist()

    # Average discounted return across all episodes
    if episode_returns:
        average_return = round(float(np.mean(episode_returns)), 4)
    else:
        average_return = 0.0

    return state_visitation_freq, state_action_counts_out, average_return