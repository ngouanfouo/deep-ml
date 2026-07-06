import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.

    Args:
        X: Input tensor of shape (batch_size, seq_len, feature_dim)
        gamma: Scale parameters of shape (1, 1, feature_dim) or broadcastable
        beta: Shift parameters of shape (1, 1, feature_dim) or broadcastable
        epsilon: Small float to avoid division by zero (default: 1e-5)

    Returns:
        np.ndarray: Normalized and scaled tensor of same shape as X
    """
    # Step 1: Compute mean and variance along the last dimension (feature_dim)
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    
    # Step 2: Normalize the features
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    
    # Step 3: Apply the learnable scale (gamma) and shift (beta) parameters
    out = gamma * X_normalized + beta
    
    return out