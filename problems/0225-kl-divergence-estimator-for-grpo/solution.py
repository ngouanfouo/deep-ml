import torch

def kl_divergence_estimator(pi_theta: torch.Tensor, pi_ref: torch.Tensor) -> torch.Tensor:
    """
    Compute the unbiased KL divergence estimator using PyTorch.
    
    Args:
        pi_theta: Current policy probabilities
        pi_ref: Reference policy probabilities
        
    Returns:
        Per-sample KL divergence estimates
    """
    # Compute the ratio of reference to policy probabilities
    # ratio = pi_ref / pi_theta
    ratio = pi_ref / pi_theta
    
    # Compute KL divergence estimator: ratio - log(ratio) - 1
    # This is the unbiased estimator used in GRPO
    kl = ratio - torch.log(ratio) - 1.0
    
    return kl