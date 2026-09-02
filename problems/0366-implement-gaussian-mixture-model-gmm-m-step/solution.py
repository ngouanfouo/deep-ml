import torch
import numpy as np

def gmm_m_step(X, gamma):
    """
    Perform the M-step of the EM algorithm for Gaussian Mixture Model.
    
    Args:
        X: Data points, shape (N, D) - torch.Tensor or array-like
        gamma: Responsibilities from E-step, shape (N, K) - torch.Tensor or array-like
    
    Returns:
        tuple: (means, covariances, mixing_coeffs)
            - means: Updated component means, shape (K, D) - torch.Tensor
            - covariances: Updated component covariances, shape (K, D, D) - torch.Tensor
            - mixing_coeffs: Updated mixing coefficients, shape (K,) - torch.Tensor
    """
    # Ensure inputs are torch Tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()
        
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma, dtype=torch.float32)
    else:
        gamma = gamma.float()
        
    N, D = X.shape
    _, K = gamma.shape
    
    # 1. Effective number of points assigned to each component (sum of responsibilities)
    N_k = gamma.sum(dim=0)  # Shape: (K,)
    
    # 2. Update means: mu_k = sum_n (gamma_nk * x_n) / N_k
    # (K, N) @ (N, D) -> (K, D)
    means = (gamma.T @ X) / torch.clamp(N_k.unsqueeze(1), min=1e-8)
    
    # 3. Update covariances: Sigma_k = sum_n gamma_nk * (x_n - mu_k)(x_n - mu_k)^T / N_k
    covariances = torch.zeros((K, D, D), dtype=X.dtype, device=X.device)
    for k in range(K):
        # Diff from mean: (N, D)
        diff = X - means[k].unsqueeze(0)
        # Weighted outer products: sum over N of gamma_nk * (diff_n outer diff_n)
        # Equivalent to: (diff * gamma[:, k:k+1]) 
        weighted_outer = (diff.unsqueeze(2) * diff.unsqueeze(1)) * gamma[:, k].view(N, 1, 1)
        covariances[k] = weighted_outer.sum(dim=0) / torch.clamp(N_k[k], min=1e-8)
        
    # Add a small regularization term to the diagonal for numerical stability (optional but robust)
    covariances += torch.eye(D, dtype=X.dtype, device=X.device) * 1e-6
    
    # 4. Update mixing coefficients: pi_k = N_k / N
    mixing_coeffs = N_k / N
    
    return means, covariances, mixing_coeffs