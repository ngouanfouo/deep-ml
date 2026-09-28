def rl_backup(backup_type: str, transitions, gamma: float, values: dict, policy: dict = None) -> float:
    """
    Compute the result of an RL backup operation.
    """

    def action_value(trans_list) -> float:
        """Expected value of taking a specific action: sum p*(r + gamma*V(s'))."""
        return sum(
            p * (r + gamma * values.get(ns, 0.0))
            for p, ns, r in trans_list
        )

    if backup_type == "action_value":
        # transitions: list of (prob, next_state, reward)
        result = action_value(transitions)

    elif backup_type == "state_value":
        # transitions: {action: [(prob, next_state, reward), ...]}
        # policy:      {action: probability}
        result = sum(
            policy[a] * action_value(trans_list)
            for a, trans_list in transitions.items()
        )

    elif backup_type == "optimal_value":
        # transitions: {action: [(prob, next_state, reward), ...]}
        result = max(
            action_value(trans_list)
            for trans_list in transitions.values()
        )

    else:
        raise ValueError(
            f"backup_type must be 'action_value', 'state_value', or 'optimal_value', got {backup_type!r}"
        )

    return round(float(result), 4)