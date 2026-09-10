import numpy as np

def fid_score(real_features, generated_features) -> float:
    """
    Compute the Frechet Inception Distance between two feature sets.

    Args:
        real_features: array-like of shape (N1, D)
        generated_features: array-like of shape (N2, D)

    Returns:
        float: the FID value.
    """
    X1 = np.array(real_features, dtype=float)
    X2 = np.array(generated_features, dtype=float)
    
    # Ensure 2D
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    
    N1, D = X1.shape
    N2 = X2.shape[0]
    
    # Compute means
    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    
    # Compute sample covariances (unbiased, divide by N-1)
    if N1 > 1:
        diff1 = X1 - mu1
        cov1 = (diff1.T @ diff1) / (N1 - 1)
    else:
        cov1 = np.zeros((D, D))
    
    if N2 > 1:
        diff2 = X2 - mu2
        cov2 = (diff2.T @ diff2) / (N2 - 1)
    else:
        cov2 = np.zeros((D, D))
    
    # Mean difference term
    diff = mu1 - mu2
    mean_term = float(diff @ diff)
    
    # Trace term: Tr(cov1 + cov2 - 2 * sqrt(cov1 @ cov2))
    # Use symmetric square root approach for numerical stability
    # Tr(sqrt(cov1 @ cov2)) = Tr(sqrt(cov1_sqrt @ cov2 @ cov1_sqrt))
    
    # Compute symmetric square root of cov1
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    eigvals1 = np.clip(eigvals1, 0, None)
    cov1_sqrt = (eigvecs1 * np.sqrt(eigvals1)) @ eigvecs1.T
    
    # Compute M = cov1_sqrt @ cov2 @ cov1_sqrt (symmetric PSD)
    M = cov1_sqrt @ cov2 @ cov1_sqrt
    
    # Eigenvalues of M
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.clip(eigvals_M, 0, None)
    
    # Tr(sqrt(cov1 @ cov2)) = sum of sqrt eigenvalues of M
    sqrt_trace = float(np.sum(np.sqrt(eigvals_M)))
    
    # Trace term
    trace_term = float(np.trace(cov1) + np.trace(cov2) - 2.0 * sqrt_trace)
    
    # FID
    fid = mean_term + trace_term
    return float(fid)