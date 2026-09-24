import numpy as np

def effective_rank(H, tol=1e-12):
    """
    Exponentiated Shannon entropy of the normalized singular values.

    Args:
        H: 2D array
        tol: singular values below this are discarded

    Returns:
        float in [1, number of retained singular values]
    """
    # Compute singular values only (no need for U, V)
    s = np.linalg.svd(H, compute_uv=False)
    
    # Discard negligible singular values
    s = s[s >= tol]
    
    if s.size == 0:
        return 0.0
    
    # Normalize to a probability distribution
    p = s / s.sum()
    
    # Shannon entropy (natural log)
    entropy = -np.sum(p * np.log(p))
    
    # Effective rank
    return float(np.exp(entropy))