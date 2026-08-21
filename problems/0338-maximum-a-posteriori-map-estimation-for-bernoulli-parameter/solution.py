import torch

def map_estimate_bernoulli(observations: torch.Tensor, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: Torch tensor of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    # Count number of successes (1s) and failures (0s)
    k = torch.sum(observations).float()  # number of successes
    n = len(observations)  # total number of observations
    failures = n - k  # number of failures
    
    # Posterior parameters: Beta(alpha + successes, beta + failures)
    posterior_alpha = alpha + k
    posterior_beta = beta + failures
    
    # Mode of Beta distribution: (alpha - 1) / (alpha + beta - 2)
    # for alpha > 1 and beta > 1
    if posterior_alpha > 1 and posterior_beta > 1:
        map_estimate = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
    elif posterior_alpha == 1 and posterior_beta > 1:
        # Mode at 0 if alpha = 1
        map_estimate = 0.0
    elif posterior_beta == 1 and posterior_alpha > 1:
        # Mode at 1 if beta = 1
        map_estimate = 1.0
    else:
        # Both parameters equal to 1: uniform distribution, any point in [0,1] is a mode
        # Return 0.5 as a reasonable default
        map_estimate = 0.5
    
    # Round to 4 decimal places and convert to float
    return round(map_estimate.item() if torch.is_tensor(map_estimate) else map_estimate, 4)