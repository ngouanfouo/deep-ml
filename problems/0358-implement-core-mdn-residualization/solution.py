import torch

def mdn_layer(f: torch.Tensor, X: torch.Tensor, sigma_inv: torch.Tensor, N: int) -> torch.Tensor:
    """
    Apply Metadata Normalization to features.

    Args:
        f: Features tensor of shape (M, D) where M is batch size, D is feature dimension
        X: Metadata matrix tensor of shape (M, K) where K is number of metadata variables
        sigma_inv: Pre-computed inverse covariance matrix tensor of shape (K, K)
        N: Total number of training samples

    Returns:
        Residualized features tensor of shape (M, D), orthogonal to metadata subspace
    """
    # Get current batch size M
    M = f.size(0)
    
    # Compute batch-level expectation E[X^T * f]
    # Shape of X^T * f is (K, D). Dividing by M gives the expectation.
    E_xf = torch.matmul(X.t(), f) / M
    
    # Calculate beta using the pre-computed sigma_inv and the population scale factor N
    # sigma_inv shape: (K, K) * E_xf shape: (K, D) -> beta shape: (K, D)
    beta = N * torch.matmul(sigma_inv, E_xf)
    
    # Compute the residualized features: r = f - X * beta
    # X shape: (M, K) * beta shape: (K, D) -> shape: (M, D)
    f_residualized = f - torch.matmul(X, beta)
    
    return f_residualized