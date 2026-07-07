import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions: KL(P || Q).
    
    Parameters:
    mu_p: mean vector of the first distribution, shape (d,)
    Cov_p: covariance matrix of the first distribution, shape (d, d)
    mu_q: mean vector of the second distribution, shape (d,)
    Cov_q: covariance matrix of the second distribution, shape (d, d)

    Returns:
    KL divergence as a float
    """
    d = mu_p.shape[0]
    
    # 1. Compute log determinants safely using Cholesky Decomposition
    # Cov = L * L^T -> det(Cov) = det(L)^2 -> log det(Cov) = 2 * sum(log(diag(L)))
    L_p = np.linalg.cholesky(Cov_p)
    L_q = np.linalg.cholesky(Cov_q)
    
    log_det_p = 2.0 * np.sum(np.log(np.diagonal(L_p)))
    log_det_q = 2.0 * np.sum(np.log(np.diagonal(L_q)))
    
    log_det_term = log_det_q - log_det_p
    
    # 2. Compute the Trace Term: Tr(Cov_q^-1 * Cov_p)
    # Solve system: Cov_q * X = Cov_p  =>  X = Cov_q^-1 * Cov_p
    inv_q_Cov_p = np.linalg.solve(Cov_q, Cov_p)
    trace_term = np.trace(inv_q_Cov_p)
    
    # 3. Compute the Mahalanobis Distance Term: (mu_q - mu_p)^T * Cov_q^-1 * (mu_q - mu_p)
    mean_diff = mu_q - mu_p
    # Solve system: Cov_q * y = mean_diff  =>  y = Cov_q^-1 * mean_diff
    inv_q_mean_diff = np.linalg.solve(Cov_q, mean_diff)
    mahalanobis_term = np.dot(mean_diff, inv_q_mean_diff)
    
    # 4. Combine all components according to the analytical formula
    kl_div = 0.5 * (log_det_term - d + trace_term + mahalanobis_term)
    
    return float(kl_div)