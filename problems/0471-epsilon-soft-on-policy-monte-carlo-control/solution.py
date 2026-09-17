import numpy as np

def epsilon_soft_mc_control(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float = 0.9,
    epsilon: float = 0.1
) -> tuple:
    """
    Epsilon-soft on-policy Monte Carlo control.
    
    Args:
        episodes: List of episodes, each is a list of (state, action, reward) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor
        epsilon: Epsilon for epsilon-soft policy
    
    Returns:
        Tuple of (Q, policy) as 2D lists rounded to 4 decimal places
    """
    # Q-table and running averages for first-visit MC
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns_sum = np.zeros((n_states, n_actions), dtype=float)
    returns_count = np.zeros((n_states, n_actions), dtype=float)

    # Initialize policy to uniform (any epsilon-soft initial policy works)
    policy = np.full((n_states, n_actions), 1.0 / n_actions, dtype=float)

    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue

        # 1. Compute the return G_t for every timestep (backward pass).
        G_values = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            _, _, r = episode[t]
            G = r + gamma * G
            G_values[t] = G

        # 2. First-visit: iterate forward, only use the first occurrence
        #    of each (state, action) pair in this episode.
        visited = set()
        visited_states = set()
        for t in range(T):
            s, a, _ = episode[t]
            if (s, a) not in visited:
                visited.add((s, a))
                visited_states.add(s)
                returns_sum[s, a] += G_values[t]
                returns_count[s, a] += 1
                Q[s, a] = returns_sum[s, a] / returns_count[s, a]

        # 3. Update the policy for every state visited in this episode
        #    to be epsilon-soft with respect to the current Q estimates.
        for s in visited_states:
            best_a = int(np.argmax(Q[s]))  # first index on ties
            for a in range(n_actions):
                if a == best_a:
                    policy[s, a] = 1.0 - epsilon + epsilon / n_actions
                else:
                    policy[s, a] = epsilon / n_actions

    Q_out = np.round(Q, 4).tolist()
    policy_out = np.round(policy, 4).tolist()

    return (Q_out, policy_out)