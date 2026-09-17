def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.
    """
    # Initialize all Q-values to 0
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        delta = 0.0
        new_Q = {}

        for s in states:
            for a in actions:
                q_value = 0.0

                for s_next in states:
                    p = transition_probs.get((s, a, s_next), 0.0)
                    if p == 0.0:
                        continue

                    r = rewards.get((s, a, s_next), 0.0)

                    # Expected value of the next state under the policy:
                    # V(s') = sum_a' pi(a'|s') * Q(s', a')
                    v_next = 0.0
                    for a_next in actions:
                        pi = policy.get((s_next, a_next), 0.0)
                        if pi != 0.0:
                            v_next += pi * Q[(s_next, a_next)]

                    q_value += p * (r + gamma * v_next)

                new_Q[(s, a)] = q_value
                delta = max(delta, abs(q_value - Q[(s, a)]))

        Q = new_Q

        if delta < theta:
            break

    # Round to 4 decimal places for the final result
    return {key: round(value, 4) for key, value in Q.items()}