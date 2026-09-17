import numpy as np

def dyna_q_planning(transitions, start_state, terminal_states, n_states, n_actions,
                     n_episodes, n_planning, alpha, gamma, epsilon, seed):
    """
    Run the Dyna-Q algorithm with planning updates on a deterministic MDP.
    """
    rng = np.random.RandomState(seed)

    # Initialize Q-table to zeros
    Q = np.zeros((n_states, n_actions), dtype=float)

    # Learned model: (state, action) -> (next_state, reward, done)
    model = {}

    steps_per_episode = []

    def epsilon_greedy(state):
        # First draw decides explore vs. exploit
        if rng.random() < epsilon:
            return int(rng.randint(n_actions))
        else:
            return int(np.argmax(Q[state]))

    def q_learning_update(state, action, next_state, reward, done):
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

    for _ in range(n_episodes):
        state = start_state
        steps = 0

        while state not in terminal_states and steps < 500:
            action = epsilon_greedy(state)

            # Real environment step
            next_state, reward, done = transitions[(state, action)]

            # Direct learning
            q_learning_update(state, action, next_state, reward, done)

            # Store in model
            model[(state, action)] = (next_state, reward, done)

            # Planning updates
            visited_pairs = list(model.keys())
            if len(visited_pairs) > 0:
                for _ in range(n_planning):
                    idx = int(rng.randint(len(visited_pairs)))
                    s_p, a_p = visited_pairs[idx]
                    ns_p, r_p, d_p = model[(s_p, a_p)]
                    q_learning_update(s_p, a_p, ns_p, r_p, d_p)

            state = next_state
            steps += 1

        steps_per_episode.append(steps)

    Q_rounded = np.round(Q, 4).tolist()

    return {
        'Q': Q_rounded,
        'steps_per_episode': steps_per_episode
    }