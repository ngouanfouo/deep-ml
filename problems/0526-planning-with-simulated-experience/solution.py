import numpy as np

def plan_with_model(Q: np.ndarray, model: dict, n_planning_steps: int, gamma: float,
                    alpha: float, seed: int = 42) -> np.ndarray:
    """
    Perform planning using simulated experience from a learned environment model.
    """
    Q = np.array(Q, dtype=float, copy=True)

    if n_planning_steps <= 0 or len(model) == 0:
        return np.round(Q, 4)

    rng = np.random.RandomState(seed)
    keys = list(model.keys())

    for _ in range(n_planning_steps):
        idx = int(rng.randint(len(keys)))
        s, a = keys[idx]
        reward, next_state = model[(s, a)]

        if next_state == -1:
            # Terminal transition: no bootstrap
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[s, a] += alpha * (target - Q[s, a])

    return np.round(Q, 4)