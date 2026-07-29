import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Convert to numpy array
    x = np.array(x, dtype=np.float64)
    
    # Compute softmax
    # For numerical stability, subtract max
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    softmax = exp_x / np.sum(exp_x)
    
    n = len(x)
    # Create Jacobian matrix
    jacobian = np.zeros((n, n), dtype=np.float64)
    
    # For softmax, the Jacobian is:
    # J[i][j] = s_i * (δ_ij - s_j)
    # where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i][j] = softmax[i] * (1 - softmax[i])
            else:
                jacobian[i][j] = -softmax[i] * softmax[j]
    
    # Alternative vectorized implementation:
    # jacobian = np.diag(softmax) - np.outer(softmax, softmax)
    # This gives the same result but is more efficient
    
    return jacobian.tolist()