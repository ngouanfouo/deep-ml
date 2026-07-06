import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    
    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics
    
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    _, C, _, _ = X.shape

    # Initialize running statistics if they are not provided
    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1))
    if running_var is None:
        running_var = np.ones((1, C, 1, 1))

    if training:
        # Step 1: Compute statistics across Batch (0), Height (2), and Width (3) axes
        # keepdims=True ensures the output shape is (1, C, 1, 1) for easy broadcasting
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)

        # Step 2: Normalize the batch
        X_norm = (X - batch_mean) / np.sqrt(batch_var + epsilon)

        # Step 3: Update running statistics using the PyTorch-style momentum formula
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
    else:
        # Inference mode: use provided running statistics instead of the batch statistics
        X_norm = (X - running_mean) / np.sqrt(running_var + epsilon)

    # Step 4: Scale and shift using the learnable parameters
    out = gamma * X_norm + beta

    return out, running_mean, running_var