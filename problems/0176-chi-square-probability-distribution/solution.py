import torch
import math

def chi_square_probability(x: float, k: int) -> float:
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x <= 0:
        return 0.0
        
    k_half = k / 2.0
    
    # Compute the probability density in log-space for numerical stability
    # log(f(x; k)) = (k/2 - 1)*log(x) - x/2 - (k/2)*log(2) - log(Gamma(k/2))
    log_prob = (
        (k_half - 1.0) * math.log(x) 
        - (x / 2.0) 
        - (k_half * math.log(2.0)) 
        - torch.lgamma(torch.tensor(k_half, dtype=torch.float64)).item()
    )
    
    probability = math.exp(log_prob)
    
    return round(probability, 3)