import numpy as np

def simulate_mdp(P, R, policy, start_state, terminal_states, gamma, num_episodes, max_steps, seed=42):
    np.random.seed(seed)
    terminal_set = set(terminal_states)

    n_states = P.shape[0]
    n_actions = P.shape[1]

    episode_returns = []

    for _ in range(num_episodes):
        state = start_state
        total_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if state in terminal_set:
                break

            action = int(np.random.choice(n_actions, p=policy[state]))
            next_state = int(np.random.choice(n_states, p=P[state, action]))

            reward = R[state, action, next_state]
            total_return += discount * reward
            discount *= gamma

            state = next_state

        # Cast to Python float before rounding so the result is a plain float
        episode_returns.append(round(float(total_return), 4))

    average_return = round(float(np.mean(episode_returns)), 4) if episode_returns else 0.0

    return episode_returns, average_return