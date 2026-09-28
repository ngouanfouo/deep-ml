import numpy as np


def mc_q_estimation(episodes: list, n_states: int, n_actions: int,
                    gamma: float, method: str = "first_visit") -> list:
    """
    Estimate Q(s, a) from episodes using Monte Carlo returns.

    Args:
        episodes: list of episodes; each episode is a list of (state, action, reward)
        n_states: number of states (0..n_states-1)
        n_actions: number of actions (0..n_actions-1)
        gamma: discount factor
        method: "first_visit" or "every_visit"

    Returns:
        Nested list of shape (n_states, n_actions), rounded to 4 decimals.
    """
    if method not in ("first_visit", "every_visit"):
        raise ValueError(
            f"method must be 'first_visit' or 'every_visit', got {method!r}"
        )

    returns_sum = np.zeros((n_states, n_actions), dtype=float)
    returns_count = np.zeros((n_states, n_actions), dtype=int)

    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue

        # --- Compute the return G_t for every timestep, working backwards ---
        G = np.zeros(T, dtype=float)
        running = 0.0
        for t in range(T - 1, -1, -1):
            _, _, r = episode[t]
            running = r + gamma * running
            G[t] = running

        if method == "first_visit":
            seen = set()
            for t in range(T):
                s, a, _ = episode[t]
                if (s, a) in seen:
                    continue
                seen.add((s, a))
                returns_sum[s, a] += G[t]
                returns_count[s, a] += 1

        else:  # every_visit
            for t in range(T):
                s, a, _ = episode[t]
                returns_sum[s, a] += G[t]
                returns_count[s, a] += 1

    # Average, leaving unvisited (s, a) pairs at 0.0
    Q = np.zeros((n_states, n_actions), dtype=float)
    visited = returns_count > 0
    Q[visited] = returns_sum[visited] / returns_count[visited]

    return np.round(Q, 4).tolist()