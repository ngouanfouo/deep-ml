import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray,
                              gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.

    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix
        R: np.ndarray of shape (n_states,), expected immediate reward per state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'

    Returns:
        np.ndarray: updated value function after n_sweeps
    """
    V_new = V.copy()                 # never mutate the caller's array
    n_states = V.shape[0]

    if method == "synchronous":
        for _ in range(n_sweeps):
            V_old = V_new.copy()     # snapshot: all states use previous sweep's values
            V_new = R + gamma * (P @ V_old)

    elif method == "in_place":
        for _ in range(n_sweeps):
            for s in range(n_states):
                # Uses the most recent V_new values for earlier states
                V_new[s] = R[s] + gamma * np.dot(P[s], V_new)

    else:
        raise ValueError(
            f"method must be 'synchronous' or 'in_place', got {method!r}"
        )

    return V_new