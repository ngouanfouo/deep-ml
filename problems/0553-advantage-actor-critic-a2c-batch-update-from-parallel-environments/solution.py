import numpy as np

def a2c_update(
    num_states: int,
    num_actions: int,
    batch_states: list,
    batch_actions: list,
    batch_rewards: list,
    batch_dones: list,
    bootstrap_states: list,
    gamma: float,
    value_coeff: float,
    entropy_coeff: float,
    alpha_theta: float,
    alpha_w: float,
    theta_init: np.ndarray,
    w_init: np.ndarray
) -> tuple:
    """
    Perform a single A2C batch update from parallel environment data.
    
    Args:
        num_states: Number of states
        num_actions: Number of actions
        batch_states: K lists of T state indices
        batch_actions: K lists of T action indices
        batch_rewards: K lists of T rewards
        batch_dones: K lists of T done flags
        bootstrap_states: K state indices for bootstrapping
        gamma: Discount factor
        value_coeff: Coefficient for value loss
        entropy_coeff: Coefficient for entropy bonus
        alpha_theta: Actor learning rate
        alpha_w: Critic learning rate
        theta_init: Initial actor params, shape (num_states, num_actions)
        w_init: Initial critic weights, shape (num_states,)
    
    Returns:
        Tuple of (w, theta) as numpy arrays
    """
    # Initialize parameters
    theta = theta_init.copy()
    w = w_init.copy()
    
    # Initialize gradient accumulators
    grad_theta = np.zeros((num_states, num_actions))
    grad_w = np.zeros(num_states)
    
    # Count total number of samples
    total_samples = 0
    
    # Process each environment's trajectory
    for states, actions, rewards, dones, bootstrap_state in zip(
        batch_states, batch_actions, batch_rewards, batch_dones, bootstrap_states
    ):
        T = len(states)
        total_samples += T
        
        # Compute returns and advantages for this trajectory
        # Bootstrapped returns: G_t = r_t + gamma * G_{t+1} (with G_T = V(bootstrap_state))
        G = w[bootstrap_state]  # Bootstrapped value at the end
        
        # We'll compute returns backwards
        returns = np.zeros(T)
        advantages = np.zeros(T)
        
        for t in reversed(range(T)):
            if dones[t]:
                # Terminal state: no bootstrapping
                G = rewards[t]
            else:
                # Non-terminal: r_t + gamma * G_{t+1}
                G = rewards[t] + gamma * G
            
            returns[t] = G
            # Advantage = G_t - V(s_t)
            advantages[t] = G - w[states[t]]
        
        # Accumulate gradients for this trajectory
        for t in range(T):
            s = states[t]
            a = actions[t]
            
            # ---- Actor gradient ----
            # Compute softmax policy for state s
            logits = theta[s, :]  # (num_actions,)
            
            # Numerically stable softmax
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)  # (num_actions,)
            
            # Policy gradient: grad_log_pi = one_hot(a) - probs
            one_hot = np.zeros(num_actions)
            one_hot[a] = 1.0
            grad_log_pi = one_hot - probs  # (num_actions,)
            
            # Accumulate policy gradient weighted by advantage
            grad_theta[s, :] += advantages[t] * grad_log_pi
            
            # ---- Entropy bonus gradient ----
            # Entropy: H = -sum(pi * log(pi))
            # Gradient of entropy w.r.t. logits: -probs_j * (log(probs_j) + H) for each action j
            
            # Compute log probabilities with numerical safety
            log_probs = np.log(probs + 1e-10)  # (num_actions,)
            
            # Compute entropy
            entropy = -np.sum(probs * log_probs)  # scalar
            
            # Entropy gradient: -probs_j * (log(probs_j) + H) for each action j
            entropy_grad = -probs * (log_probs + entropy)  # (num_actions,)
            
            # Accumulate entropy gradient
            grad_theta[s, :] += entropy_coeff * entropy_grad
            
            # ---- Critic gradient ----
            # Advantage = G_t - V(s_t) is the error signal
            grad_w[s] += advantages[t]
    
    # Average gradients over total samples
    if total_samples > 0:
        grad_theta /= total_samples
        grad_w /= total_samples
    
    # Update parameters
    theta += alpha_theta * grad_theta
    w += alpha_w * value_coeff * grad_w
    
    return w, theta