import numpy as np

def run_blocking_maze(grid_h, grid_w, walls_phase1, walls_phase2,
                      change_step, start, goal, gamma, alpha, epsilon,
                      n_planning, kappa, num_steps, seed=42):
    """
    Simulate a Dyna-Q+ agent in a blocking maze environment.
    """
    np.random.seed(seed)

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    Q = np.zeros((grid_h, grid_w, 4), dtype=float)
    model = {}                                    # (state, action) -> (reward, next_state)
    last_visit = np.zeros((grid_h, grid_w, 4), dtype=int)

    start = tuple(start)
    goal = tuple(goal)
    walls1 = set(map(tuple, walls_phase1))
    walls2 = set(map(tuple, walls_phase2))

    def walls_at(t):
        # Walls switch at the start of timestep change_step (inclusive)
        return walls2 if t >= change_step else walls1

    state = start
    cumulative_reward = 0.0
    episodes_completed = 0

    for t in range(num_steps):
        walls = walls_at(t)
        r, c = state

        # ---- Action selection: epsilon-greedy with random tie-breaking ----
        if np.random.random() < epsilon:
            action = int(np.random.randint(4))
        else:
            q = Q[r, c]
            max_q = np.max(q)
            best_actions = np.where(q == max_q)[0]
            action = int(np.random.choice(best_actions))

        # ---- Environment transition ----
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc

        if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w or (nr, nc) in walls:
            next_state = state
            reward = 0.0
            done = False
        elif (nr, nc) == goal:
            next_state = (nr, nc)
            reward = 1.0
            done = True
        else:
            next_state = (nr, nc)
            reward = 0.0
            done = False

        # ---- Q-learning update on the real step ----
        max_next = 0.0 if done else float(np.max(Q[next_state[0], next_state[1]]))
        Q[r, c, action] += alpha * (reward + gamma * max_next - Q[r, c, action])

        # ---- Model + last-visit bookkeeping ----
        model[(state, action)] = (reward, next_state)
        last_visit[r, c, action] = t

        cumulative_reward += reward

        if done:
            episodes_completed += 1
            state = start
        else:
            state = next_state

        # ---- Planning (Dyna-Q+) ----
        if model and n_planning > 0:
            keys = list(model.keys())
            n_keys = len(keys)
            for _ in range(n_planning):
                s, a = keys[np.random.randint(n_keys)]
                r_sim, s_next = model[(s, a)]
                tau = t - last_visit[s[0], s[1], a]
                bonus = kappa * np.sqrt(tau)
                target = (r_sim + bonus
                          + gamma * float(np.max(Q[s_next[0], s_next[1]])))
                Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])

    return (round(float(cumulative_reward), 4), int(episodes_completed))