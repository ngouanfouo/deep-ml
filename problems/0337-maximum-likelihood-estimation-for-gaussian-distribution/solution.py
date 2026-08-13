import torch

def gaussian_mle(data: torch.Tensor) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D torch.Tensor (or array-like) of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle) as torch.Tensor scalars
    """
    # Ensure input is a floating-point PyTorch Tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float64)
    else:
        data = data.to(dtype=torch.float64)
        
    # 1. Compute MLE Mean (Sample Mean)
    mean_mle = torch.mean(data)
    
    # 2. Compute MLE Variance (Divided by N, unbiased=False)
    variance_mle = torch.var(data, unbiased=False)
    
    return mean_mle, variance_mle