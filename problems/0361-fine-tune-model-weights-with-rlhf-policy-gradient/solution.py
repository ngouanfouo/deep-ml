import torch

def rlhf_weight_update(
    weights: torch.Tensor,
    rewards: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    log_prob_grads: torch.Tensor,
    beta: float,
    lr: float
) -> torch.Tensor:
    """
    Perform a single RLHF policy gradient weight update.

    Args:
        weights: Current model weights, shape (num_weights,)
        rewards: Rewards from reward model, shape (batch_size,)
        policy_log_probs: Log probs from current policy, shape (batch_size,)
        ref_log_probs: Log probs from reference model, shape (batch_size,)
        log_prob_grads: Gradient of log probs w.r.t. weights, shape (batch_size, num_weights)
        beta: KL penalty coefficient
        lr: Learning rate

    Returns:
        Updated weights as a torch.Tensor, shape (num_weights,)
    """
    # Compute KL divergence per sample (approximated by log probability difference)
    # KL(π_policy || π_ref) ≈ log(π_policy) - log(π_ref)
    kl_estimate = policy_log_probs - ref_log_probs
    
    # Compute adjusted rewards with KL penalty
    # Reward is penalized by how much the policy diverges from reference
    adjusted_rewards = rewards - beta * kl_estimate
    
    # Compute policy gradient: mean of adjusted_reward * grad(log_policy)
    # Reshape adjusted_rewards to (batch_size, 1) for broadcasting
    policy_gradient = torch.mean(adjusted_rewards.unsqueeze(1) * log_prob_grads, dim=0)
    
    # Update weights via gradient ascent
    updated_weights = weights + lr * policy_gradient
    
    return updated_weights