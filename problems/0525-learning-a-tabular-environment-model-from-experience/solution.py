import numpy as np
from collections import defaultdict

def learn_environment_model(experiences: list, num_states: int, num_actions: int,
                             query_state: int, query_action: int) -> tuple:
    """
    Learn a tabular environment model from experience tuples and query it.
    
    Args:
        experiences: list of (state, action, reward, next_state) tuples
        num_states: total number of states
        num_actions: total number of actions
        query_state: state to query the model for
        query_action: action to query the model for
    
    Returns:
        tuple of (transition_probs, expected_reward)
    """
    # Accumulate counts and rewards for each (s, a) pair
    transition_counts = defaultdict(lambda: defaultdict(int))
    reward_sums = defaultdict(float)
    visit_counts = defaultdict(int)

    for s, a, r, s_next in experiences:
        transition_counts[(s, a)][s_next] += 1
        reward_sums[(s, a)] += r
        visit_counts[(s, a)] += 1

    key = (query_state, query_action)

    # Unvisited state-action pair: uniform distribution, zero reward
    if visit_counts[key] == 0:
        uniform = [1.0 / num_states] * num_states
        return uniform, 0.0

    total = visit_counts[key]

    # Transition probabilities: fraction of visits to each next state
    probs = [0.0] * num_states
    for s_next, count in transition_counts[key].items():
        if 0 <= s_next < num_states:
            probs[s_next] = count / total

    # Expected reward: mean of observed rewards
    expected_reward = reward_sums[key] / total

    return probs, expected_reward