import numpy as np


def shrink_covariance(returns: np.ndarray, shrinkage: float) -> np.ndarray:
    """Linear shrinkage of the sample covariance toward a scaled identity.

    Args:
        returns (np.ndarray): (n_samples, n_assets) return matrix.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (n_assets, n_assets) shrunk covariance matrix.
    """
    returns_arr = np.asarray(returns, dtype=float)
    # Compute sample covariance with ddof=1
    S = np.cov(returns_arr, rowvar=False, ddof=1)
    
    # Handle the case where returns might be 1D or single asset if needed, 
    # but np.cov with rowvar=False on a 2D array gives (n_assets, n_assets)
    if S.ndim == 0:
        S = np.array([[S]])
        
    p = S.shape[0]
    
    # Target F = mu * I, where mu = trace(S) / p
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    
    # Shrunk covariance matrix
    sigma_shrunk = (1.0 - shrinkage) * S + shrinkage * F
    return sigma_shrunk


def min_eigenvalue(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    mat_arr = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(mat_arr)
    return float(np.min(eigenvalues))