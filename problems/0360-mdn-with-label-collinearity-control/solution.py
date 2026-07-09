import numpy as np

def mdn_with_collinearity(f: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          sigma_tilde_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply MDN with label collinearity control using an augmented design matrix.
    
    Args:
        f: Features, shape (M, D)
        X: Metadata, shape (M, K)
        y: Labels, shape (M,) or (M, 1)
        sigma_tilde_inv: Inverse covariance of augmented [X, y] matrix, shape (K+1, K+1)
        N: Total training samples
    
    Returns:
        Features with metadata (but not label) effects removed, shape (M, D)
    """
    M = f.shape[0]
    K = X.shape[1]
    
    # 1. Ensure y is a column vector (M, 1) to match the batch dimension
    if y.ndim == 1:
        y = y[:, np.newaxis]
        
    # 2. Construct the augmented design matrix Z = [X, y] of shape (M, K + 1)
    Z = np.hstack((X, y))
    
    # 3. Compute the batch-level cross-expectation E[Z^T * f] of shape (K + 1, D)
    E_zf = np.dot(Z.T, f) / M
    
    # 4. Compute joint regression coefficients beta_tilde using population scale factor N
    # beta_tilde shape: (K + 1, D)
    beta_tilde = N * np.dot(sigma_tilde_inv, E_zf)
    
    # 5. Extract only the metadata coefficients beta_X (first K rows)
    beta_X = beta_tilde[:K, :]
    
    # 6. Residualize: subtract ONLY the metadata component from the original features
    f_residualized = f - np.dot(X, beta_X)
    
    return f_residualized