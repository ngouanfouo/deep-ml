def compute_raro_rewards(
    critic_prediction: str,
    expert_position: int,
    tau_critic: float = 0.5,
    tau_policy: float = 0.5
) -> tuple[float, float]:
    """
    Compute rewards for the critic and policy in the RARO adversarial game.
    
    In RARO, a relativistic critic compares two answers (one from expert, one from policy)
    and predicts which is better. The critic and policy receive rewards based on whether
    the critic correctly identifies the expert answer.
    
    The reward structure creates an adversarial game:
    - The critic is rewarded for correctly identifying the expert answer
    - The policy is rewarded when it fools the critic into thinking its answer is better
    - Both can receive partial rewards for 'tie' predictions
    
    Args:
        critic_prediction: The critic's prediction - one of 'expert', 'policy', or 'tie'
        expert_position: Which position (1 or 2) contains the expert answer in the pair
                        (the other position contains the policy answer)
        tau_critic: Reward given to critic when it predicts 'tie' (default: 0.5)
        tau_policy: Reward given to policy when critic predicts 'tie' (default: 0.5)
    
    Returns:
        Tuple of (critic_reward, policy_reward) where:
        - critic_reward: 1.0 if critic correctly identifies expert, tau_critic if tie, 0.0 otherwise
        - policy_reward: 1.0 if critic incorrectly identifies policy as expert, tau_policy if tie, 0.0 otherwise
    
    Example:
        >>> compute_raro_rewards('expert', 1)
        (1.0, 0.0)
        >>> compute_raro_rewards('policy', 1)
        (0.0, 1.0)
        >>> compute_raro_rewards('tie', 1)
        (0.5, 0.5)
        >>> compute_raro_rewards('tie', 2, tau_critic=0.3, tau_policy=0.7)
        (0.3, 0.7)
    
    Game Theory Interpretation:
        - This is a zero-sum game when tau_critic = tau_policy = 0
        - When tau_critic = tau_policy = 0.5, tie predictions share the reward equally
        - Adjusting tau parameters allows tuning the reward distribution for ambiguous cases
    """
    # Validation
    valid_predictions = {'expert', 'policy', 'tie'}
    if critic_prediction not in valid_predictions:
        raise ValueError(f"critic_prediction must be one of {valid_predictions}, got '{critic_prediction}'")
    
    if expert_position not in {1, 2}:
        raise ValueError(f"expert_position must be 1 or 2, got {expert_position}")
    
    # Clamp tau values to [0, 1] range
    tau_critic = max(0.0, min(1.0, tau_critic))
    tau_policy = max(0.0, min(1.0, tau_policy))
    
    # Determine ground truth based on expert position
    # If expert_position=1, then position 1 is expert, position 2 is policy
    # If expert_position=2, then position 2 is expert, position 1 is policy
    expert_at_position = expert_position
    policy_at_position = 1 if expert_position == 2 else 2
    
    # Reward mapping based on critic prediction
    # The critic gets rewarded when it says 'expert' and the expert is actually there
    # The policy gets rewarded when the critic says 'policy' (mistaking policy for expert)
    if critic_prediction == 'expert':
        critic_reward = 1.0
        policy_reward = 0.0
        
    elif critic_prediction == 'policy':
        critic_reward = 0.0
        policy_reward = 1.0
        
    else:  # 'tie'
        critic_reward = tau_critic
        policy_reward = tau_policy
    
    return critic_reward, policy_reward