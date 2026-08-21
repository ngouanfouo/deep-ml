import torch

def distance_correlation_squared(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute squared distance correlation between X and Y.
    
    Args:
        X: Tensor of shape (n_samples,) or (n_samples, n_features)
        Y: Tensor of shape (n_samples,) or (n_samples, n_features)
    
    Returns:
        dcor^2 value between 0 and 1
    """
    # Ensure inputs are 2D tensors
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)
    
    n = X.shape[0]
    
    # Compute pairwise distance matrices
    # For X: compute all pairwise Euclidean distances
    # Using broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)
    X_diff = X.unsqueeze(1) - X.unsqueeze(0)  # (n, n, d)
    X_dist = torch.sqrt(torch.sum(X_diff ** 2, dim=2))  # (n, n)
    
    Y_diff = Y.unsqueeze(1) - Y.unsqueeze(0)  # (n, n, d)
    Y_dist = torch.sqrt(torch.sum(Y_diff ** 2, dim=2))  # (n, n)
    
    # Double-center the distance matrices
    # A_ij = a_ij - row_mean_i - col_mean_j + grand_mean
    X_row_mean = torch.mean(X_dist, dim=1, keepdim=True)
    X_col_mean = torch.mean(X_dist, dim=0, keepdim=True)
    X_grand_mean = torch.mean(X_dist)
    X_centered = X_dist - X_row_mean - X_col_mean + X_grand_mean
    
    Y_row_mean = torch.mean(Y_dist, dim=1, keepdim=True)
    Y_col_mean = torch.mean(Y_dist, dim=0, keepdim=True)
    Y_grand_mean = torch.mean(Y_dist)
    Y_centered = Y_dist - Y_row_mean - Y_col_mean + Y_grand_mean
    
    # Compute distance covariance: dCov^2 = (1/n^2) * sum(A_ij * B_ij)
    dCov_sq = torch.mean(X_centered * Y_centered)
    
    # Compute distance variances
    dVarX_sq = torch.mean(X_centered * X_centered)
    dVarY_sq = torch.mean(Y_centered * Y_centered)
    
    # Compute distance correlation squared
    # Handle numerical precision issues
    if dVarX_sq <= 0 or dVarY_sq <= 0:
        return 0.0
    
    dCor_sq = dCov_sq / torch.sqrt(dVarX_sq * dVarY_sq)
    
    # Clamp to [0, 1] due to numerical precision
    dCor_sq = torch.clamp(dCor_sq, 0.0, 1.0)
    
    return dCor_sq.item()