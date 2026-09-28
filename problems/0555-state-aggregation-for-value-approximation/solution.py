import numpy as np


def state_aggregation_mc(
    episodes: list,
    n_states: int,
    group_assignments: list,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using state aggregation with Monte Carlo returns.

    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
        n_states: Total number of states (states are integers 0 to n_states-1).
        group_assignments: List of length n_states mapping each state to its group.
        gamma: Discount factor.

    Returns:
        V: Estimated value for each state as numpy array of shape (n_states,).
    """
    group_assignments = np.asarray(group_assignments)

    # Accumulate returns per group
    group_returns = {}

    for episode in episodes:
        T = len(episode)
        running = 0.0
        for t in range(T - 1, -1, -1):
            s, r = episode[t]
            running = r + gamma * running
            g = int(group_assignments[s])
            group_returns.setdefault(g, []).append(running)

    # Group value = mean of all returns assigned to that group
    group_values = {g: float(np.mean(rs)) for g, rs in group_returns.items()}

    # Each state inherits its group's value; unvisited groups stay 0.0
    V = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        g = int(group_assignments[s])
        V[s] = group_values.get(g, 0.0)

    return V