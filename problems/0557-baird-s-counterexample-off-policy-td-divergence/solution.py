import numpy as np


def bairds_counterexample(w_init: list, alpha: float, gamma: float, update_states: list) -> list:
    """
    Run semi-gradient off-policy TD(0) updates on Baird's counterexample.

    Args:
        w_init: Initial weight vector of length 8
        alpha: Learning rate (step size)
        gamma: Discount factor
        update_states: List of state indices (0-6) from which the solid
                       action is taken, processed in order

    Returns:
        Updated weight vector as a list of floats
    """
    # ---- Feature map ----
    def phi(s: int) -> np.ndarray:
        f = np.zeros(8, dtype=float)
        if s == 6:
            f[6] = 1.0
            f[7] = 2.0
        else:
            f[s] = 2.0
            f[7] = 1.0
        return f

    # ---- Run updates ----
    w = np.array(w_init, dtype=float)
    rho = 7.0          # importance sampling ratio for solid action
    reward = 0.0
    next_state = 6

    for s in update_states:
        phi_s = phi(s)
        phi_next = phi(next_state)

        v_s = float(phi_s @ w)
        v_next = float(phi_next @ w)

        td_error = reward + gamma * v_next - v_s
        w = w + alpha * rho * td_error * phi_s

    return [float(x) for x in w]