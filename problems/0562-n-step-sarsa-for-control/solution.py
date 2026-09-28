import numpy as np

def n_step_sarsa_control(env: dict, terminal_states: set, start_state: int,
                         num_states: int, num_actions: int, n: int,
                         alpha: float, gamma: float, epsilon: float,
                         num_episodes: int, seed: int = 42) -> list:
    """
    Implement n-step Sarsa for on-policy control (Sutton & Barto pseudocode).
    """
    rng = np.random.RandomState(seed)
    Q = np.zeros((num_states, num_actions), dtype=float)

    def epsilon_greedy(state: int) -> int:
        if rng.random() < epsilon:
            return int(rng.randint(num_actions))
        return int(np.argmax(Q[state]))  # ties → smallest index

    for _ in range(num_episodes):
        S = start_state
        A = epsilon_greedy(S)

        # Trajectory buffers (1-indexed rewards; rewards[0] is a dummy)
        states = [S]
        actions = [A]
        rewards = [0.0]

        T = float('inf')
        t = 0

        while True:
            if t < T:
                if (S, A) in env:
                    S_next, R = env[(S, A)]
                else:
                    S_next, R = S, 0.0
                states.append(S_next)
                rewards.append(R)

                if S_next in terminal_states:
                    T = t + 1
                else:
                    A = epsilon_greedy(S_next)
                    actions.append(A)

                S = S_next

            # Perform the update for time step tau = t - n + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                upper = int(min(tau + n, T))
                for i in range(tau + 1, upper + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                # Bootstrap if the episode hasn't terminated within n steps
                if tau + n < T:
                    G += (gamma ** n) * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            if tau >= T - 1:
                break
            t += 1

    return np.round(Q, 4).tolist()