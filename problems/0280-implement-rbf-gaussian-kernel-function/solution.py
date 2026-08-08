import torch

def rbf_kernel(X1: torch.Tensor, X2: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    The RBF kernel is defined as: K(x, y) = exp(-gamma * ||x - y||^2)
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Compute squared Euclidean distances efficiently using matrix operations
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    
    # Compute squared norms for X1: (n1, 1)
    X1_sq = torch.sum(X1 ** 2, dim=1, keepdim=True)  # (n1, 1)
    
    # Compute squared norms for X2: (1, n2)
    X2_sq = torch.sum(X2 ** 2, dim=1, keepdim=True).T  # (1, n2)
    
    # Compute dot product between X1 and X2: (n1, n2)
    # Use torch.mm for matrix multiplication
    dot_product = torch.mm(X1, X2.T)  # (n1, n2)
    
    # Compute squared distances: (n1, n2)
    # Broadcasting: (n1, 1) + (1, n2) - 2*(n1, n2)
    squared_distances = X1_sq + X2_sq - 2 * dot_product
    
    # Clamp negative values to zero (due to numerical precision)
    squared_distances = torch.clamp(squared_distances, min=0.0)
    
    # Compute RBF kernel: exp(-gamma * squared_distances)
    kernel_matrix = torch.exp(-gamma * squared_distances)
    
    return kernel_matrix