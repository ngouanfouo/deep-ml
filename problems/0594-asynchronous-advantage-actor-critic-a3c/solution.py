import numpy as np

def a3c_update(
    worker_trajectories: list,
    policy_params: list,
    value_params: list,
    n_steps: int,
    gamma: float,
    alpha_policy: float,
    alpha_value: float,
    entropy_coeff: float
) -> dict:
    """
    Simulate the A3C update across multiple workers.

    Args:
        worker_trajectories: List of episodes, each a list of (state, action, reward) tuples.
        policy_params: 2D list of shape (num_states, num_actions) - softmax logits.
        value_params: 1D list of shape (num_states,) - tabular value estimates.
        n_steps: Number of steps for n-step returns.
        gamma: Discount factor.
        alpha_policy: Policy learning rate.
        alpha_value: Value function learning rate.
        entropy_coeff: Entropy regularization coefficient.

    Returns:
        Dictionary with 'policy_params' and 'value_params'.
    """
    # Convert parameters to numpy arrays to facilitate inplace mathematical updates
    policy_params = np.array(policy_params, dtype=float)
    value_params = np.array(value_params, dtype=float)
    
    num_actions = policy_params.shape[1]
    
    for trajectory in worker_trajectories:
        T = len(trajectory)
        
        # Initialize accumulated local gradients for this worker trajectory
        d_policy = np.zeros_like(policy_params)
        d_value = np.zeros_like(value_params)
        
        # Compute targets and advantages for each time step t in the trajectory
        for t in range(T):
            state_t, action_t, _ = trajectory[t]
            
            # 1. Compute the n-step return G_t
            G = 0.0
            horizon = min(t + n_steps, T)
            
            # Accumulate discounted rewards within the window
            for k in range(t, horizon):
                G += (gamma ** (k - t)) * trajectory[k][2]
                
            # Bootstrap if the lookahead window does not hit the terminal state boundary
            if t + n_steps < T:
                state_horizon = trajectory[t + n_steps][0]
                G += (gamma ** n_steps) * value_params[state_horizon]
                
            # 2. Compute Advantage metric
            advantage = G - value_params[state_t]
            
            # 3. Calculate Policy Metrics (Softmax probabilities and entropy)
            logits = policy_params[state_t]
            shift_logits = logits - np.max(logits)
            exp_logits = np.exp(shift_logits)
            probs = exp_logits / np.sum(exp_logits)
            
            # Logarithmic probability for cross-entropy check
            epsilon = 1e-15
            log_probs = np.log(probs + epsilon)
            
            # 4. Score Function Gradient for the action taken: d(log pi(a|s)) / d(theta)
            # The gradient with respect to logit index is: I(a == i) - pi(i | s)
            grad_log_pi = -probs.copy()
            grad_log_pi[action_t] += 1.0
            
            # 5. Calculate Entropy Regularization Gradient
            # H = -sum(p * log(p)). The gradient with respect to logit i is:
            # -p(i) * (log p(i) + 1) - (-p(i) * sum(p * (log p + 1))) = -p(i) * (log p(i) - sum(p * log p))
            mean_log_prob = np.dot(probs, log_probs)
            grad_entropy = -probs * (log_probs - mean_log_prob)
            
            # Combine the advantage signal weight and the exploration bonus
            policy_grad_t = grad_log_pi * advantage + entropy_coeff * grad_entropy
            
            # Accumulate local step gradients
            d_policy[state_t] += policy_grad_t
            d_value[state_t] += (G - value_params[state_t])
            
        # Apply asynchronous local gradients to global parameters at the end of the trajectory
        policy_params += alpha_policy * d_policy
        value_params += alpha_value * d_value

    return {
        'policy_params': np.round(policy_params, 4).tolist(),
        'value_params': np.round(value_params, 4).tolist()
    }