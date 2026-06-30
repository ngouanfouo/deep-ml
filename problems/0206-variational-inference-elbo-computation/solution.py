import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior distribution p(z)
        prior_std: Standard deviation of prior distribution p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        ELBO value (float)
    """
    # Sample z from variational distribution q(z) ~ N(q_mean, q_std^2)
    z_samples = np.random.normal(q_mean, q_std, n_samples)
    
    # Convert x to numpy array for vectorized operations
    x = np.array(x)
    
    # Compute expected log-likelihood: E_q[log p(x|z)]
    # For each sample z, compute log p(x|z) = sum over data points of log N(x_i | z, likelihood_std^2)
    # Then average over samples
    
    # Log-likelihood for each sample: log p(x|z) = -0.5 * sum((x - z)^2) / likelihood_std^2 - 0.5 * n * log(2π * likelihood_std^2)
    n_data = len(x)
    log_likelihoods = np.array([
        -0.5 * np.sum((x - z)**2) / (likelihood_std**2) - 0.5 * n_data * np.log(2 * np.pi * likelihood_std**2)
        for z in z_samples
    ])
    expected_log_likelihood = np.mean(log_likelihoods)
    
    # Compute expected log-prior: E_q[log p(z)]
    # log p(z) = -0.5 * (z - prior_mean)^2 / prior_std^2 - 0.5 * log(2π * prior_std^2)
    log_priors = np.array([
        -0.5 * (z - prior_mean)**2 / (prior_std**2) - 0.5 * np.log(2 * np.pi * prior_std**2)
        for z in z_samples
    ])
    expected_log_prior = np.mean(log_priors)
    
    # Compute entropy of variational distribution: H[q] = 0.5 * log(2πe * q_std^2)
    entropy = 0.5 * np.log(2 * np.pi * np.e * q_std**2)
    
    # ELBO = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z)]
    # Note: E_q[log q(z)] = -H[q]
    # So ELBO = E_q[log p(x|z)] + E_q[log p(z)] + H[q]
    elbo = expected_log_likelihood + expected_log_prior + entropy
    
    return float(elbo)