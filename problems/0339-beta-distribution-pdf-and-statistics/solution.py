import torch
from torch.distributions import Beta

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics using PyTorch.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance' as torch.Tensor scalars
    """
    # Create Beta distribution
    dist = Beta(torch.tensor(alpha), torch.tensor(beta_param))
    
    # Check if x is within (0, 1)
    if x <= 0 or x >= 1:
        pdf = torch.tensor(0.0)
    else:
        # Evaluate PDF at x
        pdf = dist.log_prob(torch.tensor(x)).exp()
    
    # Compute mean and variance
    mean = dist.mean
    variance = dist.variance
    
    return {
        'pdf': pdf,
        'mean': mean,
        'variance': variance
    }