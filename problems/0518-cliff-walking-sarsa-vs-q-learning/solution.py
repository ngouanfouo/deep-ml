import numpy as np

def cliff_walking_comparison(num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42):
    """
    Run SARSA and Q-Learning on the Cliff Walking gridworld.
    """
    height, width = 4, 12
    start = (3, 0)
    goal = (3, 11)
    cliff = set((3, c) for c in range(1, 11))

    # Action deltas: 0=up, 1=down, 2=left, 3=right
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def epsilon_greedy(Q, state, rng):
        if rng.random() < epsilon:
            return int(rng.randint(4))
        else:
            return int(np.argmax(Q[state]))

    def step(state, action):
        dr, dc = deltas[action]
        r, c = state
        nr = max(0, min(height - 1, r + dr))
        nc = max(0, min(width - 1, c + dc))
        return (nr, nc)

    # ---------- SARSA ----------
    def run_sarsa():
        rng = np.random.RandomState(seed)
        Q = np.zeros((height, width, 4), dtype=float)
        total_rewards = []

        for _ in range(num_episodes):
            state = start
            action = epsilon_greedy(Q, state, rng)
            episode_reward = 0.0
            steps = 0

            while state != goal and steps < 1000:
                next_state = step(state, action)

                if next_state in cliff:
                    reward = -100.0
                    next_state = start
                    # Terminal? No—cliff resets to start, episode continues.
                    # But the transition is over; get next action from next state.
                    next_action = epsilon_greedy(Q, next_state, rng)
                    target = reward + gamma * Q[next_state][next_action]
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = next_state
                    action = next_action
                elif next_state == goal:
                    reward = -1.0
                    target = reward  # terminal, no bootstrap
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = next_state
                else:
                    reward = -1.0
                    next_action = epsilon_greedy(Q, next_state, rng)
                    target = reward + gamma * Q[next_state][next_action]
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = next_state
                    action = next_action

                episode_reward += reward
                steps += 1

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    # ---------- Q-Learning ----------
    def run_qlearning():
        rng = np.random.RandomState(seed)
        Q = np.zeros((height, width, 4), dtype=float)
        total_rewards = []

        for _ in range(num_episodes):
            state = start
            episode_reward = 0.0
            steps = 0

            while state != goal and steps < 1000:
                if rng.random() < epsilon:
                    action = int(rng.randint(4))
                else:
                    action = int(np.argmax(Q[state]))

                next_state = step(state, action)

                if next_state in cliff:
                    reward = -100.0
                    target = reward  # max Q over next state, but reset to start
                    # Actually Q-learning: target = reward + gamma * max Q(next_state, ·)
                    target = reward + gamma * np.max(Q[next_state])
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = start
                elif next_state == goal:
                    reward = -1.0
                    target = reward
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = next_state
                else:
                    reward = -1.0
                    target = reward + gamma * np.max(Q[next_state])
                    Q[state][action] += alpha * (target - Q[state][action])
                    state = next_state

                episode_reward += reward
                steps += 1

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    sarsa_avg = run_sarsa()
    qlearning_avg = run_qlearning()

    return (round(sarsa_avg, 1), round(qlearning_avg, 1))