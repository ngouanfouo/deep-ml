import numpy as np

def retrace_targets(
    rewards: np.ndarray, 
    Q_values: np.ndarray, 
    expected_next_Q: np.ndarray, 
    target_pi: np.ndarray, 
    behavior_mu: np.ndarray, 
    dones: np.ndarray, 
    gamma: float, 
    lam: float
) -> np.ndarray:
    """
    Compute Retrace(lambda) targets for a trajectory of experience.
    
    Args:
        rewards: (T,) rewards at each step
        Q_values: (T,) Q(s_k, a_k) estimates
        expected_next_Q: (T,) E_pi[Q(s_{k+1}, .)] (0 if terminal)
        target_pi: (T,) target policy action probabilities
        behavior_mu: (T,) behavior policy action probabilities
        dones: (T,) 1.0 if episode terminates, 0.0 otherwise
        gamma: discount factor
        lam: trace decay parameter
    
    Returns:
        (T,) array of Retrace targets, rounded to 4 decimal places
    """
    T = len(rewards)
    targets = np.zeros(T, dtype=float)
    
    # Step 1: Compute trace coefficients c_t = lambda * min(1, pi_t / mu_t)
    # These represent the correction coefficients for the action taken at time t
    c = lam * np.minimum(1.0, target_pi / behavior_mu)
    
    # Step 2: Compute temporal difference errors
    # delta_t = r_t + gamma * (1 - done_t) * E[Q_next] - Q_t
    deltas = rewards + gamma * (1.0 - dones) * expected_next_Q - Q_values
    
    # Step 3: Run backward recursion to compile targets across the sequence
    # For the last element in the trajectory array, there is no subsequent step:
    targets[T - 1] = Q_values[T - 1] + deltas[T - 1]
    
    for t in range(T - 2, -1, -1):
        if dones[t] == 1.0:
            # Episode boundaries clip or cut the trace forward sequence
            targets[t] = Q_values[t] + deltas[t]
        else:
            # Standard retrace backward structural relation equation
            # Notice we use the coefficient c[t+1] corresponding to the next step
            targets[t] = Q_values[t] + deltas[t] + gamma * c[t + 1] * (targets[t + 1] - Q_values[t + 1])
            
    return np.round(targets, 4)