import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: Input tensor of shape (B, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        num_groups: Number of groups to split the channels into
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized, scaled, and shifted tensor of shape (B, C, H, W)
    """
    B, C, H, W = X.shape
    assert C % num_groups == 0, "Channels must be evenly divisible by num_groups"
    
    # 1. Reshape to split channels into groups: (B, G, C//G, H, W)
    channels_per_group = C // num_groups
    X_reshaped = X.reshape(B, num_groups, channels_per_group, H, W)
    
    # 2. Compute mean and variance across (C//G, H, W) -> axes (2, 3, 4)
    # Keepdims=True preserves shape for broadcasting during subtraction/division
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # 3. Normalize the tensor within each group
    X_norm = (X_reshaped - mean) / np.sqrt(var + epsilon)
    
    # 4. Reshape back to original dimensions (B, C, H, W) to apply gamma and beta
    X_norm = X_norm.reshape(B, C, H, W)
    
    # 5. Reshape gamma and beta to (1, C, 1, 1) for broadcasting over B, H, and W
    gamma_broadcast = gamma.reshape(1, C, 1, 1)
    beta_broadcast = beta.reshape(1, C, 1, 1)
    
    # 6. Apply linear transformation
    out = X_norm * gamma_broadcast + beta_broadcast
    
    return out