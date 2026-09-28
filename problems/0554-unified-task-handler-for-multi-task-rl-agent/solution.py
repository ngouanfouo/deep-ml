import numpy as np


def unified_task_handler(state_features: np.ndarray,
                         task_weights: np.ndarray,
                         task_id: int,
                         epsilon: float = 0.0) -> dict:
    """
    Compute task-conditioned action values and epsilon-greedy action probabilities.

    Args:
        state_features: (num_actions, feature_dim) feature vectors for each action
        task_weights: (num_tasks, feature_dim) weight vectors for each task
        task_id: integer index of the current task
        epsilon: exploration parameter for epsilon-greedy (0 <= epsilon <= 1)

    Returns:
        dict with keys:
          - 'q_values': numpy array of shape (num_actions,)
          - 'greedy_action': int
          - 'action_probs': numpy array of shape (num_actions,)
    """
    state_features = np.asarray(state_features, dtype=float)
    task_weights = np.asarray(task_weights, dtype=float)

    num_actions = state_features.shape[0]

    # Task-conditioned Q-values: linear value approximation per action
    w_task = task_weights[task_id]                  # (feature_dim,)
    q_values = state_features @ w_task              # (num_actions,)

    # Greedy action: highest Q-value; np.argmax returns the lowest index on ties
    greedy_action = int(np.argmax(q_values))

    # Epsilon-greedy action probabilities
    if epsilon <= 0.0:
        action_probs = np.zeros(num_actions, dtype=float)
        action_probs[greedy_action] = 1.0
    else:
        action_probs = np.full(num_actions, epsilon / num_actions, dtype=float)
        action_probs[greedy_action] += (1.0 - epsilon)

    return {
        'q_values': q_values,
        'greedy_action': greedy_action,
        'action_probs': action_probs,
    }