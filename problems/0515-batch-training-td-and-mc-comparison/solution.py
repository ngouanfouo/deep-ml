import numpy as np

def batch_td_mc_comparison(
    episodes: list,
    n_states: int,
    gamma: float,
    alpha: float,
    max_iterations: int
) -> tuple:
    """
    Compare batch TD(0) and batch every-visit MC on a fixed dataset.
    """
    # ---------- Batch every-visit Monte Carlo ----------
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states

    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue
        G_values = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            _, reward = episode[t]
            G = reward + gamma * G
            G_values[t] = G
        for t in range(T):
            state, _ = episode[t]
            returns_sum[state] += G_values[t]
            returns_count[state] += 1

    V_mc = [0.0] * n_states
    for s in range(n_states):
        if returns_count[s] > 0:
            V_mc[s] = returns_sum[s] / returns_count[s]

    # ---------- Batch TD(0) ----------
    # Collect every transition (s, r, s') from the batch, with s' = -1 meaning terminal.
    # Group transitions by their originating state.
    transitions_by_state = [[] for _ in range(n_states)]
    for episode in episodes:
        T = len(episode)
        for t in range(T):
            state, reward = episode[t]
            if t + 1 < T:
                next_state = episode[t + 1][0]
            else:
                next_state = -1  # terminal
            transitions_by_state[state].append((reward, next_state))

    V_td = [0.0] * n_states

    for _ in range(max_iterations):
        new_V = V_td[:]
        delta = 0.0
        for s in range(n_states):
            trans = transitions_by_state[s]
            if not trans:
                continue
            # Average the TD target over all transitions from s
            total_target = 0.0
            for reward, next_state in trans:
                v_next = 0.0 if next_state == -1 else V_td[next_state]
                total_target += reward + gamma * v_next
            mean_target = total_target / len(trans)

            updated = V_td[s] + alpha * (mean_target - V_td[s])
            new_V[s] = updated
            delta = max(delta, abs(updated - V_td[s]))
        V_td = new_V
        if delta < 1e-12:
            break

    V_td_rounded = [round(v, 4) for v in V_td]
    V_mc_rounded = [round(v, 4) for v in V_mc]

    return V_td_rounded, V_mc_rounded