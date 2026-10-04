def rl_backup_diagram(
    transitions: dict,
    policy: list,
    state: int,
    backup_type: str,
    V: list,
    gamma: float
) -> dict:
    """
    Generate a structured backup diagram for RL value estimation.
    
    Args:
        transitions: Dict mapping (state, action) -> list of (prob, next_state, reward)
        policy: 2D list where policy[s][a] = probability of action a in state s
        state: The state to compute the backup for
        backup_type: 'v_expectation' or 'v_optimal'
        V: List of current state value estimates
        gamma: Discount factor
    
    Returns:
        Dictionary with keys: action_values, aggregation, backed_up_value, diagram
    """
    # Identify available actions for the given state
    available_actions = sorted(
        action for (s, action) in transitions.keys() if s == state
    )
    
    action_values = {}
    diagram = []
    
    for action in available_actions:
        # Compute action value Q(state, action)
        q_value = 0.0
        transition_details = []
        
        for prob, next_state, reward in transitions[(state, action)]:
            target = reward + gamma * V[next_state]
            contribution = prob * target
            q_value += contribution
            
            transition_details.append({
                "next_state": next_state,
                "prob": round(prob, 4),
                "reward": round(reward, 4),
                "target": round(target, 4),
                "contribution": round(contribution, 4)
            })
        
        q_value_rounded = round(q_value, 4)
        action_values[action] = q_value_rounded
        
        # Build diagram entry for this action
        diagram.append({
            "action": action,
            "policy_prob": round(policy[state][action], 4),
            "transitions": transition_details,
            "action_value": q_value_rounded
        })
    
    # Aggregate action values into a backed-up state value
    if backup_type == "v_expectation":
        aggregation = "expectation"
        backed_up_value = sum(
            policy[state][action] * action_values[action]
            for action in available_actions
        )
    elif backup_type == "v_optimal":
        aggregation = "maximization"
        backed_up_value = max(action_values.values()) if action_values else 0.0
    else:
        raise ValueError("backup_type must be 'v_expectation' or 'v_optimal'")
    
    backed_up_value = round(backed_up_value, 4)
    
    return {
        "action_values": action_values,
        "aggregation": aggregation,
        "backed_up_value": backed_up_value,
        "diagram": diagram
    }