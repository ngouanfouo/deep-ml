import numpy as np

def efficient_binary_td_update(initial_weights: list, transitions: list, alpha: float, gamma: float) -> np.ndarray:
    """
    Perform TD(0) updates with binary feature representations.
    """
    weights = np.array(initial_weights, dtype=float)

    for active_s, reward, active_s_next, done in transitions:
        # Current estimate V(s) = sum of weights at active features
        v_s = float(np.sum(weights[active_s])) if len(active_s) > 0 else 0.0

        # Next-state estimate V(s'), 0 if terminal
        if done or len(active_s_next) == 0:
            v_s_next = 0.0
        else:
            v_s_next = float(np.sum(weights[active_s_next]))

        # TD error
        td_error = reward + gamma * v_s_next - v_s

        # Semi-gradient update: only active features of s
        if len(active_s) > 0:
            weights[active_s] += alpha * td_error

    return weights