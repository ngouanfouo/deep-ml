import numpy as np

def actor_critic_step(state_features, action, reward, next_state_features, done,
                      theta, w, gamma, alpha_theta, alpha_w):
    """
    Perform one step of the Actor-Critic algorithm with linear function approximation.
    
    Args:
        state_features: np.ndarray shape (n_features,) - feature vector for current state
        action: int - action taken
        reward: float - reward received
        next_state_features: np.ndarray shape (n_features,) - feature vector for next state
        done: bool - whether episode terminated
        theta: np.ndarray shape (n_features, n_actions) - actor parameters
        w: np.ndarray shape (n_features,) - critic weight vector
        gamma: float - discount factor
        alpha_theta: float - actor learning rate
        alpha_w: float - critic learning rate
    
    Returns:
        tuple: (updated_theta, updated_w, td_error, action_probs)
            - updated_theta: np.ndarray shape (n_features, n_actions)
            - updated_w: np.ndarray shape (n_features,)
            - td_error: float - the temporal difference error
            - action_probs: np.ndarray shape (n_actions,) - softmax probabilities before update
    """
    # ----- Critic: Compute TD error -----
    # V(s) = w^T * phi(s)
    V_s = np.dot(w, state_features)
    
    # V(s') = w^T * phi(s') if not done, else 0
    if done:
        V_s_next = 0.0
    else:
        V_s_next = np.dot(w, next_state_features)
    
    # TD error: delta = r + gamma * V(s') - V(s)
    td_error = reward + gamma * V_s_next - V_s
    
    # ----- Critic: Update critic weights (semi-gradient TD(0)) -----
    # w = w + alpha_w * delta * phi(s)
    w_new = w + alpha_w * td_error * state_features
    
    # ----- Actor: Compute action probabilities (before update) -----
    # Logits = phi(s)^T * theta
    logits = np.dot(state_features, theta)  # shape (n_actions,)
    
    # Numerically stable softmax
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    action_probs = exp_logits / np.sum(exp_logits)  # shape (n_actions,)
    
    # ----- Actor: Compute gradient of log-policy -----
    # For softmax policy, gradient = phi(s) * (one_hot(action) - probs)^T
    # This gives shape (n_features, n_actions)
    one_hot = np.zeros_like(action_probs)
    one_hot[action] = 1.0
    grad_log_pi = np.outer(state_features, one_hot - action_probs)
    
    # ----- Actor: Update actor parameters -----
    # theta = theta + alpha_theta * delta * grad_log_pi
    theta_new = theta + alpha_theta * td_error * grad_log_pi
    
    return theta_new, w_new, td_error, action_probs