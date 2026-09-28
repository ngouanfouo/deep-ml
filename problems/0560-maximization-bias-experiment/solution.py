import numpy as np

def maximization_bias_experiment(n_episodes: int, n_actions_B: int, epsilon: float,
                                 alpha: float, gamma: float, mean_B: float,
                                 std_B: float, seed: int) -> tuple:
    """
    Run Q-learning and Double Q-learning on the maximization bias MDP.
    """
    # ============================================================
    # Q-Learning
    # ============================================================
    rng_q = np.random.RandomState(seed)
    Q_A = np.zeros(2)
    Q_B = np.zeros(n_actions_B)
    left_count_q = 0

    for _ in range(n_episodes):
        # --- State A: epsilon-greedy action selection ---
        if rng_q.random() < epsilon:
            a = rng_q.randint(2)
        else:
            a = int(np.argmax(Q_A))

        if a == 0:  # left
            left_count_q += 1
            # Update Q_A[0] with target = 0 + gamma * max(Q_B)
            Q_A[0] += alpha * (gamma * np.max(Q_B) - Q_A[0])

            # --- State B: epsilon-greedy action selection ---
            if rng_q.random() < epsilon:
                b = rng_q.randint(n_actions_B)
            else:
                b = int(np.argmax(Q_B))

            r = rng_q.normal(mean_B, std_B)
            # Terminal transition: target = r + gamma * 0
            Q_B[b] += alpha * (r - Q_B[b])

        else:  # right -> terminal, reward 0
            Q_A[1] += alpha * (0.0 - Q_A[1])

    q_left_pct = round(left_count_q / n_episodes, 2)

    # ============================================================
    # Double Q-Learning
    # ============================================================
    rng_dq = np.random.RandomState(seed + 1)
    Q1_A = np.zeros(2); Q2_A = np.zeros(2)
    Q1_B = np.zeros(n_actions_B); Q2_B = np.zeros(n_actions_B)
    left_count_dq = 0

    for _ in range(n_episodes):
        # --- State A: action selection uses Q1_A + Q2_A ---
        if rng_dq.random() < epsilon:
            a = rng_dq.randint(2)
        else:
            a = int(np.argmax(Q1_A + Q2_A))

        if a == 0:  # left
            left_count_dq += 1
            # Coin flip: update Q1_A or Q2_A
            if rng_dq.random() < 0.5:
                best_b = int(np.argmax(Q1_B))
                target = gamma * Q2_B[best_b]
                Q1_A[0] += alpha * (target - Q1_A[0])
            else:
                best_b = int(np.argmax(Q2_B))
                target = gamma * Q1_B[best_b]
                Q2_A[0] += alpha * (target - Q2_A[0])

            # --- State B: action selection uses Q1_B + Q2_B ---
            if rng_dq.random() < epsilon:
                b = rng_dq.randint(n_actions_B)
            else:
                b = int(np.argmax(Q1_B + Q2_B))

            r = rng_dq.normal(mean_B, std_B)

            # Coin flip: update Q1_B or Q2_B
            if rng_dq.random() < 0.5:
                Q1_B[b] += alpha * (r - Q1_B[b])
            else:
                Q2_B[b] += alpha * (r - Q2_B[b])

        else:  # right -> terminal, reward 0
            if rng_dq.random() < 0.5:
                Q1_A[1] += alpha * (0.0 - Q1_A[1])
            else:
                Q2_A[1] += alpha * (0.0 - Q2_A[1])

    double_q_left_pct = round(left_count_dq / n_episodes, 2)

    return (q_left_pct, double_q_left_pct)