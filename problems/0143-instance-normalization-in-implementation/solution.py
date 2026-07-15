import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    B, C, H, W = X.shape
    
    # Compute mean across spatial dimensions (H, W) for each instance and channel
    # Keep dimensions for broadcasting: (B, C, 1, 1)
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    
    # Compute variance across spatial dimensions
    var = np.var(X, axis=(2, 3), keepdims=True)
    
    # Normalize: (X - mean) / sqrt(var + epsilon)
    # Gamma and beta are of shape (C,), need to reshape for broadcasting: (1, C, 1, 1)
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)
    
    # Apply normalization and then scale and shift
    out = gamma_reshaped * (X - mean) / np.sqrt(var + epsilon) + beta_reshaped
    
    return out