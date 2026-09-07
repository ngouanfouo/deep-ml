import numpy as np


def combine_alphas(mu: np.ndarray, cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Mean-variance weights from shrunk covariance, normalised to sum to 1.

    Args:
        mu (np.ndarray): (p,) expected returns of each alpha.
        cov (np.ndarray): (p, p) covariance matrix of the alphas.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (p,) weights summing to 1.
    """
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    
    p = cov_arr.shape[0]
    
    # Compute the shrinkage target F = mu_bar * I, where mu_bar = trace(cov) / p
    mu_bar = np.trace(cov_arr) / p
    F = mu_bar * np.eye(p)
    
    # Shrunk covariance matrix
    sigma_shrunk = (1.0 - shrinkage) * cov_arr + shrinkage * F
    
    # Solve for unnormalised weights: sigma_shrunk @ w_raw = mu
    w_raw = np.linalg.solve(sigma_shrunk, mu_arr)
    
    # Normalise weights to sum to 1
    w = w_raw / np.sum(w_raw)
    return w