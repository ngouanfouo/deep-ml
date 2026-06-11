import numpy as np

def reinforce_with_baseline(episode, theta, w, gamma, alpha_theta, alpha_w):
    """
    Perform one episode update of REINFORCE with a value function baseline.
    
    Args:
        episode: list of (state, action, reward) tuples
        theta: np.ndarray of shape (n_states, n_actions), policy parameters
        w: np.ndarray of shape (n_states,), value function parameters
        gamma: float, discount factor
        alpha_theta: float, policy learning rate
        alpha_w: float, value function learning rate
    
    Returns:
        tuple: (theta_new, w_new) updated parameters
    """
    # Create copies to avoid mutating original parameters in place
    theta_new = np.array(theta, dtype=float)
    w_new = np.array(w, dtype=float)
    
    T = len(episode)
    if T == 0:
        return theta_new, w_new
        
    # 1. Compute the discounted returns G_t for each timestep
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G
        
    # 2. Process each timestep sequentially from t = 0 to T - 1
    for t in range(T):
        state, action, _ = episode[t]
        G_t = returns[t]
        
        # Compute softmax action probabilities for the current state
        # Subtracting np.max ensures numerical stability for exponentials
        logits = theta_new[state, :]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Compute the advantage delta = G_t - V(S_t)
        advantage = G_t - w_new[state]
        
        # Update value parameters online
        w_new[state] += alpha_w * advantage
        
        # Compute the score function (gradient of log pi(A_t | S_t))
        # grad = 1 - prob for the chosen action, and -prob for all other actions
        grad_log_pi = -probs
        grad_log_pi[action] += 1.0
        
        # Update policy parameters scaled by policy learning rate, gamma^t, and advantage
        discount = gamma ** t
        theta_new[state, :] += alpha_theta * discount * advantage * grad_log_pi
        
    return theta_new, w_new

# --- Verification matching your exact example ---
