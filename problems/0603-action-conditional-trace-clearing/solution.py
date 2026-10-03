import numpy as np

def action_conditional_trace_episode(Q: np.ndarray, episode: list, gamma: float, alpha: float, lam: float) -> np.ndarray:
    """
    Process a single episode with eligibility traces and action-conditional clearing.
    """
    Q = np.array(Q, dtype=float)
    n_states, n_actions = Q.shape
    e = np.zeros_like(Q)

    for (state, action, reward, next_state, done, next_action_greedy) in episode:
        # 1. TD error: use max Q at next state, or 0 for terminal
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(Q[next_state])
        delta = td_target - Q[state, action]

        # 2. Accumulating trace for the current state-action pair
        e[state, action] += 1.0

        # 3. Apply the TD update to all Q-values, weighted by traces
        Q += alpha * delta * e

        # 4. Conditionally decay or clear traces
        if done:
            e[:] = 0.0
        elif next_action_greedy:
            e *= gamma * lam
        else:
            e[:] = 0.0

    return np.round(Q, 4)