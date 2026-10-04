import numpy as np

def softmax_policy_gradient_step(
    theta: np.ndarray,
    states: list,
    actions: list,
    weights: list,
    alpha: float
) -> tuple:
    """
    Perform one policy gradient update step using softmax action selection.
    
    Args:
        theta: np.ndarray of shape (num_states, num_actions) - action preference parameters
        states: list of int - state index for each sample
        actions: list of int - action taken for each sample
        weights: list of float - scalar weight (return/advantage) for each sample
        alpha: float - learning rate
    
    Returns:
        tuple: (updated_theta, action_probs_list)
            - updated_theta: np.ndarray of shape (num_states, num_actions)
            - action_probs_list: list of np.ndarray, softmax probabilities for each sample's state
    """
    theta = np.array(theta, dtype=float)
    num_states, num_actions = theta.shape

    grad = np.zeros_like(theta)
    action_probs_list = []

    n = len(states)
    if n == 0:
        return theta.copy(), action_probs_list

    for s, a, w in zip(states, actions, weights):
        # Numerically stable softmax for the sample's state
        logits = theta[s]
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)

        action_probs_list.append(probs.copy())

        # Score function: one-hot(action) - probs
        score = -probs
        score[a] += 1.0

        # Accumulate weighted gradient for this state
        grad[s] += w * score

    # Average gradient over all samples
    grad /= n

    # Apply policy gradient update
    updated_theta = theta + alpha * grad

    return updated_theta, action_probs_list