def greedy_policy_improvement(V: dict, transitions: dict, gamma: float) -> tuple:
    """
    Perform greedy policy improvement given a value function and MDP model.
    
    Args:
        V: dict mapping state -> float (state-value function)
        transitions: dict mapping (state, action) -> list of (probability, next_state, reward)
        gamma: float, discount factor
    
    Returns:
        tuple: (policy, Q)
            policy: dict mapping state -> best action
            Q: dict mapping (state, action) -> float
    """
    # 1. Compute Q(s, a) for every (state, action) pair in transitions.
    Q = {}
    for (s, a), outcomes in transitions.items():
        q_value = 0.0
        for prob, next_state, reward in outcomes:
            q_value += prob * (reward + gamma * V.get(next_state, 0.0))
        Q[(s, a)] = q_value

    # 2. Greedy policy: for each state, pick the action with max Q,
    #    breaking ties in lexicographic (alphabetical) order.
    #    We collect candidate actions per state first, then sort them
    #    so the lexicographically smallest action is considered first.
    actions_by_state = {}
    for (s, a) in Q.keys():
        actions_by_state.setdefault(s, []).append(a)

    policy = {}
    for s, actions in actions_by_state.items():
        # Sort actions alphabetically so that, in a tie, the first one encountered
        # with a strictly greater Q-value wins; equal Q-values keep the earlier action.
        best_action = None
        best_q = None
        for a in sorted(actions):
            q = Q[(s, a)]
            if best_q is None or q > best_q:
                best_q = q
                best_action = a
        policy[s] = best_action

    return policy, Q