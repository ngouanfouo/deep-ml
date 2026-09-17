def random_walk_td_mc(episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1) -> tuple:
    """
    Run TD(0) and first-visit Monte Carlo prediction on random walk episodes.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of non-terminal states
        gamma: Discount factor
        alpha: Learning rate for TD(0)
    
    Returns:
        Tuple of (td_values, mc_values) as lists of floats
    """
    # ---------- TD(0) ----------
    V_td = [0.0] * n_states
    for episode in episodes:
        for state, reward, next_state, done in episode:
            if done:
                # Terminal transition: no bootstrap
                target = reward
            else:
                target = reward + gamma * V_td[next_state]
            V_td[state] += alpha * (target - V_td[state])

    # ---------- First-visit Monte Carlo ----------
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states

    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue

        # Compute the return G_t for every timestep by working backward
        G = 0.0
        G_values = [0.0] * T
        for t in reversed(range(T)):
            _, reward, _, _ = episode[t]
            G = reward + gamma * G
            G_values[t] = G

        # Only the first occurrence of each state in this episode counts
        visited = set()
        for t in range(T):
            state = episode[t][0]
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G_values[t]
                returns_count[state] += 1

    # Average returns; unvisited states stay at 0.0
    V_mc = [0.0] * n_states
    for s in range(n_states):
        if returns_count[s] > 0:
            V_mc[s] = returns_sum[s] / returns_count[s]

    return V_td, V_mc