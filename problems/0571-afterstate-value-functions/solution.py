def afterstate_value_iteration(
    states: list,
    actions: dict,
    afterstate_map: dict,
    env_dynamics: dict,
    gamma: float,
    theta: float = 0.0001,
    max_iterations: int = 1000
) -> tuple:
    """
    Perform value iteration using afterstate value functions.

    Afterstate value update (synchronous):
        V(a) = sum_{(p, r, s') in env_dynamics[a]} p * (r + gamma * V_state[s'])
    where
        V_state[s'] = max_{a'} V(afterstate_map[(s', a')])  if non-terminal
                    = 0                                    if terminal
    """
    afterstates = set(afterstate_map.values())
    V = {a: 0.0 for a in afterstates}

    for _ in range(max_iterations):
        # State values from the *current* afterstate values
        V_state = {}
        for s in states:
            acts = actions.get(s, [])
            if not acts:
                V_state[s] = 0.0
            else:
                V_state[s] = max(V[afterstate_map[(s, a)]] for a in acts)

        # Synchronous update of all afterstate values
        V_new = {}
        for a in afterstates:
            total = 0.0
            for p, r, s_next in env_dynamics.get(a, []):
                total += p * (r + gamma * V_state[s_next])
            V_new[a] = total

        # Compute delta for reporting / early stop
        delta = max(abs(V_new[a] - V[a]) for a in afterstates) if afterstates else 0.0
        V = V_new

        # Tight convergence check: with a gamma-contraction, ||V - V*|| can be
        # up to delta/(1 - gamma). Only stop once that bound is far below the
        # requested precision.
        if delta < theta * (1.0 - gamma) * 1e-2:
            break

    # Greedy policy
    policy = {}
    for s in states:
        acts = actions.get(s, [])
        if not acts:
            continue
        best_action = None
        best_value = float('-inf')
        for a in sorted(acts):                       # lexicographically smallest wins on ties
            val = V[afterstate_map[(s, a)]]
            if val > best_value:
                best_value = val
                best_action = a
        policy[s] = best_action

    afterstate_values = {a: round(v, 4) for a, v in V.items()}
    return afterstate_values, policy