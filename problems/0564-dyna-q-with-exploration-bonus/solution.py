import numpy as np

def dyna_q_plus_planning(
    Q: np.ndarray,
    model: dict,
    tau: np.ndarray,
    current_step: int,
    kappa: float,
    alpha: float,
    gamma: float,
    planning_samples: list
) -> np.ndarray:
    """
    Perform Dyna-Q+ planning steps with exploration bonus.

    For each (s, a) in planning_samples:
        r, s' = model[(s, a)]
        bonus = kappa * sqrt(current_step - tau[s, a])
        target = r + bonus + gamma * max_a' Q[s', a']
        Q[s, a] += alpha * (target - Q[s, a])
    """
    Q = Q.copy()  # do not mutate the caller's array

    for (s, a) in planning_samples:
        reward, next_state = model[(s, a)]

        # Exploration bonus: grows with time since last real visit
        bonus = kappa * np.sqrt(current_step - tau[s, a])

        # Q-learning-style target using augmented reward
        target = reward + bonus + gamma * np.max(Q[next_state])

        Q[s, a] += alpha * (target - Q[s, a])

    return np.round(Q, 4)