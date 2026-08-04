import numpy as np

def mean_ablate(activations: np.ndarray, mask: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Apply mean ablation to node activations.
    
    Args:
        activations: Original node activations, shape (n,) or (batch, n)
        mask: Binary mask where 1 = ablate (replace with mean), 0 = keep original
        means: Precomputed mean activations for each node, shape (n,)
    
    Returns:
        Ablated activations with same shape as input
    """
    # Convert mask to boolean
    mask_bool = mask.astype(bool)
    
    # If activations is 1D
    if activations.ndim == 1:
        return np.where(mask_bool, means, activations)
    else:
        # If activations is 2D (batch, n), broadcast means to match shape
        # Expand means to (1, n) and broadcast it across batches
        means_expanded = means.reshape(1, -1)
        # Use np.where with broadcasting
        # mask_bool is (n,), we need to expand it to (1, n) for broadcasting
        mask_expanded = mask_bool.reshape(1, -1)
        return np.where(mask_expanded, means_expanded, activations)