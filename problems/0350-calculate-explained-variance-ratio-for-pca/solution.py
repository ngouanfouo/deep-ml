import torch

def explained_variance_ratio(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features) as a torch.Tensor
    
    Returns:
        torch.Tensor of explained variance ratios sorted in descending order
    """
    # Center the data
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean
    
    # Compute the covariance matrix (unbiased estimator: divide by n-1)
    n_samples = X_centered.shape[0]
    covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Compute eigenvalues (and eigenvectors, though eigenvectors are not needed)
    eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    # Ensure eigenvalues are non-negative (handle numerical issues)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    # Calculate explained variance ratios
    total_variance = torch.sum(eigenvalues)
    explained_ratios = eigenvalues / total_variance
    
    return explained_ratios