import torch

def compute_policy_gradient(theta: torch.Tensor, episodes: list[list[tuple[int, int, float]]]) -> torch.Tensor:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    num_states, num_actions = theta.shape
    total_gradient = torch.zeros_like(theta)
    
    # Count total number of episodes for averaging
    num_episodes = len(episodes)
    
    for episode in episodes:
        # Step 1: Compute returns (discounted sum of rewards)
        # For REINFORCE with no discount, return at time t = sum of rewards from t to end
        T = len(episode)
        returns = torch.zeros(T)
        
        # Compute cumulative rewards from the end
        cumulative_return = 0.0
        for t in reversed(range(T)):
            cumulative_return += episode[t][2]  # add reward at time t
            returns[t] = cumulative_return
        
        # Step 2: Compute gradient for each time step
        for t, (state, action, _) in enumerate(episode):
            # Compute softmax policy for this state
            logits = theta[state]  # shape (num_actions,)
            # Softmax: exp(logits) / sum(exp(logits))
            # For numerical stability, subtract max
            logits_stable = logits - torch.max(logits)
            exp_logits = torch.exp(logits_stable)
            probs = exp_logits / torch.sum(exp_logits)
            
            # Compute gradient of log π(a|s) with respect to theta
            # ∇_θ log π(a|s) = ∇_θ (θ_{s,a} - log(sum(exp(θ_s))))
            # = e_a - π(s)
            grad_log_policy = torch.zeros_like(theta)
            grad_log_policy[state] = torch.eye(num_actions)[action] - probs
            
            # Multiply by return and accumulate
            total_gradient += grad_log_policy * returns[t]
    
    # Average over episodes
    average_gradient = total_gradient / num_episodes
    
    return average_gradient