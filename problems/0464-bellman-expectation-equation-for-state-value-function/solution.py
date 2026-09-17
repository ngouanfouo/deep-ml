import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    num_states, num_actions, _ = P.shape

    # P_pi[s, t] = sum_a policy[s, a] * P[s, a, t]
    P_pi = np.einsum('sa,sat->st', policy, P)

    # R_pi[s] = sum_a sum_t policy[s, a] * P[s, a, t] * R[s, a, t]
    R_pi = np.einsum('sa,sat,sat->s', policy, P, R)

    # Solve (I - gamma * P_pi) V = R_pi
    A = np.eye(num_states) - gamma * P_pi
    V = np.linalg.solve(A, R_pi)

    return V