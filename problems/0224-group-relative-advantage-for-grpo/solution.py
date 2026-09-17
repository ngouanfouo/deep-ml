import torch

def compute_group_relative_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute the Group Relative Advantage for GRPO using PyTorch.
    
    Args:
        rewards: 1D tensor of rewards for a group of outputs
        
    Returns:
        1D tensor of normalized advantages
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)

    mean = rewards.mean()
    # Population std (unbiased=False) so that for rewards like [0,1,0,1]
    # with mean 0.5 the std is exactly 0.5, matching GRPO's normalization.
    std = rewards.std(unbiased=False)

    # Epsilon guards against division by zero when all rewards in the group
    # are identical (std = 0), in which case all advantages become 0.
    eps = 1e-8
    advantages = (rewards - mean) / (std + eps)
    return advantages