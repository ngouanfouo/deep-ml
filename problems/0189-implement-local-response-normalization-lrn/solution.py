import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size (number of neighboring channels to sum over)
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """
    N, C, H, W = x.shape
    
    # 1. Compute the squared activations component: (a_{x,y}^j)^2
    squared_x = x ** 2
    
    # 2. Track window radius bounds based on standard half-window truncation
    # PyTorch and AlexNet define the half-window range as [i - n//2, i + n//2]
    half_n = n // 2
    
    # Pre-allocate the running channel sum accumulation matrix
    sum_sq = np.zeros_like(squared_x)
    
    # 3. Accumulate values within the sliding neighborhood channel slice window
    for i in range(C):
        start = max(0, i - half_n)
        end = min(C, i + half_n + 1)
        sum_sq[:, i, :, :] = np.sum(squared_x[:, start:end, :, :], axis=1)
        
    # 4. Apply scale normalization transformations: (k + alpha * sum_sq) ** beta
    denom = (k + (alpha / n) * sum_sq) ** beta
    
    return x / denom