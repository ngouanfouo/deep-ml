import numpy as np

def tsne_gradient(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the t-SNE cost function.
    
    Args:
        P: (n, n) symmetric numpy array of joint probabilities in high-dimensional space
        Y: (n, d) numpy array of current low-dimensional embedding
    
    Returns:
        gradient: (n, d) numpy array of gradients for each point, rounded to 4 decimal places
    """
    n, d = Y.shape
    
    # Compute pairwise squared distances in low-dimensional space
    # Using broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d) -> sum over last dim
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, n, d)
    squared_distances = np.sum(diff ** 2, axis=2)  # (n, n)
    
    # Compute Q distribution using Student's t-distribution with 1 degree of freedom
    # q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum_{k!=l} (1 + ||y_k - y_l||^2)^(-1)
    # But we'll compute the gradient formula directly without normalizing denominator
    # For the gradient, we use the unnormalized version and then normalize
    
    # Compute kernel values: 1 / (1 + squared_distances)
    kernel = 1.0 / (1.0 + squared_distances)
    
    # Set diagonal to 0 (since q_ii = 0)
    np.fill_diagonal(kernel, 0.0)
    
    # Normalize to get Q (joint probabilities)
    Q = kernel / np.sum(kernel)
    
    # Compute the gradient
    # Formula: dC/dY_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)
    # More precisely: 4 * sum_j (p_ij - q_ij) * q_ij * (y_i - y_j)
    # Since q_ij = (1 + ||y_i - y_j||^2)^(-1) / Z, the gradient becomes:
    # 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
    
    # Alternatively, using the standard t-SNE gradient formula:
    # dC/dY_i = 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
    # But here q_ij already includes the normalization, so we use:
    # dC/dY_i = 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
    
    # Compute factor for each pair: (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1)
    # Note: q_ij already has the normalization included
    factor = (P - Q) * kernel  # (n, n)
    
    # Compute gradient for each point
    gradient = np.zeros((n, d))
    for i in range(n):
        # For each point, sum over j: factor[i,j] * (y_i - y_j)
        grad_i = np.sum((factor[i, :, np.newaxis] * (Y[i] - Y)), axis=0)
        gradient[i] = 4.0 * grad_i
    
    # Round to 4 decimal places
    return np.round(gradient, 4)