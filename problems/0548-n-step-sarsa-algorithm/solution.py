import numpy as np


def n_step_sarsa(n_states: int, n_actions: int, transitions: dict,
                 terminal_states: list, n: int, gamma: float, alpha: float,
                 epsilon: float, num_episodes: int, seed: int = 42) -> list:
    """
    n-Step Sarsa for on-policy TD control (online updates, Sutton & Barto).
    """
    np.random.seed(seed)
    Q = np.zeros((n_states, n_actions), dtype=float)
    terminal_set = set(terminal_states)
    non_terminal = [s for s in range(n_states) if s not in terminal_set]

    def epsilon_greedy(state: int) -> int:
        """Epsilon-greedy; ties in argmax go to the smallest action index."""
        if np.random.random() < epsilon:
            return int(np.random.randint(0, n_actions))
        return int(np.argmax(Q[state]))

    for _ in range(num_episodes):
        # Random non-terminal start
        state = int(np.random.choice(non_terminal))
        action = epsilon_greedy(state)

        # Trajectory buffers: states[t], actions[t], rewards[t] with
        # rewards[t] = R_{t} (rewards[0] is a dummy placeholder)
        states = [state]
        actions = [action]
        rewards = [0.0]

        T = float('inf')
        t = 0

        while True:
            if t < T:
                next_state, reward = transitions[(state, action)]
                states.append(next_state)
                rewards.append(reward)
                if next_state in terminal_set:
                    T = t + 1
                else:
                    next_action = epsilon_greedy(next_state)
                    actions.append(next_action)
                    state = next_state
                    action = next_action

            tau = t - n + 1
            if tau >= 0:
                # n-step return
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                # Bootstrap only if the episode hasn't terminated within n steps
                if tau + n < T:
                    G += (gamma ** n) * Q[states[tau + n], actions[tau + n]]

                s_tau = states[tau]
                a_tau = actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau >= T - 1:
                break
            t += 1

    return np.round(Q, 4).tolist()