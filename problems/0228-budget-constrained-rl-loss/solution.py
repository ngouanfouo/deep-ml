import numpy as np

def rl_budget_loss(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    response_lengths: np.ndarray,
    token_budget: int,
    kl_coef: float,
    budget_penalty_coef: float
) -> float:
    """
    Compute the budget-constrained RL loss.
    
    The loss combines:
    1. Budget penalty for responses exceeding token_budget
    2. Advantage estimation (adjusted reward - baseline)
    3. KL regularization between current and old policy
    
    Loss formula: E[(advantage - kl_term)^2]
    
    Args:
        rewards: Shape (batch_size, K) - rewards for K samples per prompt
        log_probs: Shape (batch_size, K) - log π_θ(y|x) current policy
        old_log_probs: Shape (batch_size, K) - log π_old(y|x) old policy
        response_lengths: Shape (batch_size, K) - token lengths of responses
        token_budget: Maximum allowed tokens before penalty
        kl_coef: τ coefficient for KL regularization
        budget_penalty_coef: λ coefficient for budget penalty
        
    Returns:
        Scalar loss value (float)
    """
    # Convert to numpy arrays if needed
    rewards = np.array(rewards, dtype=np.float64)
    log_probs = np.array(log_probs, dtype=np.float64)
    old_log_probs = np.array(old_log_probs, dtype=np.float64)
    response_lengths = np.array(response_lengths, dtype=np.float64)
    
    # Step 1: Apply budget penalty
    # Penalty = -budget_penalty_coef * max(0, length - token_budget)
    over_budget = np.maximum(0, response_lengths - token_budget)
    budget_penalty = -budget_penalty_coef * over_budget
    
    # Adjust rewards with budget penalty
    adjusted_rewards = rewards + budget_penalty
    
    # Step 2: Compute baseline (mean over samples for each prompt)
    # In GRPO style, baseline is the mean reward per group
    # For each prompt (row), compute mean of adjusted rewards across samples
    baseline = np.mean(adjusted_rewards, axis=1, keepdims=True)
    
    # Step 3: Compute advantages (adjusted reward - baseline)
    advantages = adjusted_rewards - baseline
    
    # Step 4: Compute KL divergence terms
    # kl_term = kl_coef * (log_probs - old_log_probs)
    kl_term = kl_coef * (log_probs - old_log_probs)
    
    # Step 5: Compute squared loss
    # Loss = E[(advantage - kl_term)^2]
    squared_diff = (advantages - kl_term) ** 2
    loss = np.mean(squared_diff)
    
    return float(loss)