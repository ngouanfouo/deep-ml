import numpy as np

def windy_gridworld_sarsa(height: int, width: int, start: tuple, goal: tuple, wind: list,
                          num_episodes: int, alpha: float = 0.5, gamma: float = 1.0,
                          epsilon: float = 0.1, seed: int = 42) -> tuple:
    """
    Run SARSA on a windy gridworld environment.
    """
    np.random.seed(seed)

    # Movement deltas for actions: 0=Up, 1=Right, 2=Down, 3=Left
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Q-table
    Q = np.zeros((height, width, 4), dtype=float)

    def epsilon_greedy_action(row, col):
        if np.random.random() < epsilon:
            return int(np.random.randint(4))
        else:
            return int(np.argmax(Q[row, col]))

    def step(row, col, action):
        dr, dc = deltas[action]
        new_row = row + dr
        new_col = col + dc
        # Wind pushes upward (toward row 0) based on the column we moved into.
        # Here we apply the wind based on the current column, as stated in the task.
        new_row -= wind[col] if 0 <= col < width else 0
        # Clip to grid bounds
        new_row = max(0, min(height - 1, new_row))
        new_col = max(0, min(width - 1, new_col))
        return new_row, new_col

    episode_lengths = []

    for _ in range(num_episodes):
        row, col = start
        action = epsilon_greedy_action(row, col)
        steps = 0

        while (row, col) != goal and steps < 10000:
            next_row, next_col = step(row, col, action)
            reward = -1.0

            if (next_row, next_col) == goal:
                # Terminal transition: no bootstrap
                target = reward
                Q[row, col, action] += alpha * (target - Q[row, col, action])
                row, col = next_row, next_col
                steps += 1
                break
            else:
                next_action = epsilon_greedy_action(next_row, next_col)
                target = reward + gamma * Q[next_row, next_col, next_action]
                Q[row, col, action] += alpha * (target - Q[row, col, action])
                row, col = next_row, next_col
                action = next_action
                steps += 1

        episode_lengths.append(steps)

    # Greedy policy grid (first index on ties)
    policy = np.argmax(Q, axis=2).astype(int)

    return Q, policy, episode_lengths