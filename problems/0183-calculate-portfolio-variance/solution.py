import torch

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio using PyTorch.
    """
    # Convert inputs to tensors
    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    # Input validation
    if cov_tensor.ndim != 2 or cov_tensor.shape[0] != cov_tensor.shape[1]:
        raise ValueError("Covariance matrix must be a 2D square matrix.")
    if weights_tensor.ndim != 1:
        raise ValueError("Weights must be a 1D vector.")
    if cov_tensor.shape[0] != weights_tensor.shape[0]:
        raise ValueError("Covariance matrix dimensions must match the length of weights.")
    
    # Calculate portfolio variance: w^T * Sigma * w
    variance = torch.matmul(weights_tensor, torch.matmul(cov_tensor, weights_tensor))
    
    return float(variance.item())