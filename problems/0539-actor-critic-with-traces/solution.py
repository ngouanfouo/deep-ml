import numpy as np

def actor_critic_traces(
    num_states: int,
    num_actions: int,
    episode: list,
    gamma: float,
    lambda_w: float,
    lambda_theta: float,
    alpha_w: float,
    alpha_theta: float,
    w_init: np.ndarray,
    theta_init: np.ndarray
) -> tuple:
    """
    Episodic actor-critic with eligibility traces.
    
    Args:
        num_states: Number of states
        num_actions: Number of actions
        episode: List of (state, action, reward, next_state, done) tuples
        gamma: Discount factor
        lambda_w: Trace decay for critic
        lambda_theta: Trace decay for actor
        alpha_w: Critic learning rate
        alpha_theta: Actor learning rate
        w_init: Initial critic weights, shape (num_states,)
        theta_init: Initial actor parameters, shape (num_states, num_actions)
    
    Returns:
        Tuple of (w, theta) as numpy arrays
    """
    # Initialize weights and traces
    w = w_init.copy()
    theta = theta_init.copy()
    z_w = np.zeros(num_states)  # Critic eligibility trace
    z_theta = np.zeros((num_states, num_actions))  # Actor eligibility trace
    I = 1.0  # Discount accumulator for actor trace
    
    # Process each transition in the episode
    for state, action, reward, next_state, done in episode:
        # ----- Compute TD error -----
        # V(s) = w[s]
        V_s = w[state]
        
        # V(s') = w[next_state] if not done, else 0
        if done:
            V_s_next = 0.0
        else:
            V_s_next = w[next_state]
        
        # TD error: delta = r + gamma * V(s') - V(s)
        delta = reward + gamma * V_s_next - V_s
        
        # ----- Critic: Update eligibility trace -----
        # z_w = gamma * lambda_w * z_w + one_hot(state)
        z_w = gamma * lambda_w * z_w
        z_w[state] += 1.0
        
        # ----- Actor: Compute policy and policy gradient -----
        # Get logits for current state
        logits = theta[state, :]  # shape (num_actions,)
        
        # Numerically stable softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)  # shape (num_actions,)
        
        # Policy gradient: one_hot(action) - probs
        one_hot = np.zeros(num_actions)
        one_hot[action] = 1.0
        grad_log_pi = one_hot - probs  # shape (num_actions,)
        
        # ----- Actor: Update eligibility trace -----
        # z_theta = gamma * lambda_theta * z_theta + I * grad_log_pi at state
        # Note: grad_log_pi is per state, so we add to the state row
        z_theta = gamma * lambda_theta * z_theta
        z_theta[state, :] += I * grad_log_pi
        
        # Update discount accumulator: I = gamma * I
        I = gamma * I
        
        # ----- Update critic weights -----
        # w = w + alpha_w * delta * z_w
        w += alpha_w * delta * z_w
        
        # ----- Update actor parameters -----
        # theta = theta + alpha_theta * delta * z_theta
        theta += alpha_theta * delta * z_theta
    
    return w, theta