import numpy as np

def compute_fvd(features_real, features_gen) -> float:
    """
    Compute the Frechet Video Distance between two sets of feature vectors.

    Args:
        features_real: array-like of shape (N, D) -- features of real clips.
        features_gen:  array-like of shape (M, D) -- features of generated clips.

    Returns:
        float: FVD score (>= 0).
    """
    X = np.array(features_real, dtype=float)
    Y = np.array(features_gen, dtype=float)
    
    # Ensure 2D (handle D=1 as column vectors)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    N, D = X.shape
    M = Y.shape[0]
    
    # Sample means
    mu_r = X.mean(axis=0)
    mu_g = Y.mean(axis=0)
    
    # Sample covariances (unbiased, divide by n-1)
    if N > 1:
        diff_r = X - mu_r
        cov_r = (diff_r.T @ diff_r) / (N - 1)
    else:
        cov_r = np.zeros((D, D))
    cov_r = (cov_r + cov_r.T) / 2.0  # symmetrize against fp noise
    
    if M > 1:
        diff_g = Y - mu_g
        cov_g = (diff_g.T @ diff_g) / (M - 1)
    else:
        cov_g = np.zeros((D, D))
    cov_g = (cov_g + cov_g.T) / 2.0
    
    # Term 1: squared L2 distance between means
    diff = mu_r - mu_g
    mean_term = float(diff @ diff)
    
    # Term 2: trace of (cov_r + cov_g - 2 * sqrt(cov_r @ cov_g))
    # Compute symmetric square root of cov_r via eigendecomposition
    eigvals_r, eigvecs_r = np.linalg.eigh(cov_r)
    eigvals_r = np.clip(eigvals_r, 0, None)  # clip fp-negative eigenvalues
    cov_r_sqrt = (eigvecs_r * np.sqrt(eigvals_r)) @ eigvecs_r.T
    
    # M = cov_r_sqrt @ cov_g @ cov_r_sqrt  (symmetric PSD)
    M_mat = cov_r_sqrt @ cov_g @ cov_r_sqrt
    M_mat = (M_mat + M_mat.T) / 2.0  # symmetrize for eigh
    
    # Tr(sqrt(cov_r @ cov_g)) = sum of sqrt eigenvalues of M_mat
    eigvals_M = np.linalg.eigvalsh(M_mat)
    eigvals_M = np.clip(eigvals_M, 0, None)  # clip fp-negative eigenvalues
    sqrt_trace = float(np.sum(np.sqrt(eigvals_M)))
    
    trace_term = float(np.trace(cov_r) + np.trace(cov_g) - 2.0 * sqrt_trace)
    
    fvd = mean_term + trace_term
    
    # FVD should be non-negative; clip tiny negative values from fp noise
    return float(max(fvd, 0.0))