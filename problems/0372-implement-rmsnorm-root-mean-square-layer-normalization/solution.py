import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    
    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        g   : np.ndarray of shape (features,) - gain parameter
        eps : float - small constant for numerical stability
    
    Returns:
        np.ndarray of same shape as x
    """
    # 1. Compute the mean of squared values along the feature dimension (axis -1)
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    
    # 2. Compute the Root Mean Square (RMS) with epsilon for numerical stability
    rms = np.sqrt(mean_sq + eps)
    
    # 3. Normalize the input by dividing by RMS
    normalized = x / rms
    
    # 4. Scale by the learnable gain parameter g (broadcasted across the batch)
    output = normalized * g
    
    return output