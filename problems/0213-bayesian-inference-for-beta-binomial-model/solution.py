import torch

def bayesian_inference_beta_binomial(prior_alpha: float, prior_beta: float, 
                                     successes: int, trials: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Bayesian inference for Beta-Binomial model.
    
    Args:
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
        successes: Number of successes observed
        trials: Total number of trials
    
    Returns:
        Tuple of (posterior_alpha, posterior_beta, posterior_mean) as tensors where:
        - posterior_alpha: Updated alpha parameter
        - posterior_beta: Updated beta parameter
        - posterior_mean: Mean of posterior distribution
    """
    # Convert to tensors
    prior_alpha = torch.tensor(prior_alpha, dtype=torch.float64)
    prior_beta = torch.tensor(prior_beta, dtype=torch.float64)
    successes = torch.tensor(successes, dtype=torch.float64)
    trials = torch.tensor(trials, dtype=torch.float64)
    
    # Calculate failures
    failures = trials - successes
    
    # For Beta-Binomial conjugacy:
    # Posterior: Beta(alpha_prior + successes, beta_prior + failures)
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    
    # Posterior mean of Beta distribution = alpha / (alpha + beta)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    return posterior_alpha, posterior_beta, posterior_mean