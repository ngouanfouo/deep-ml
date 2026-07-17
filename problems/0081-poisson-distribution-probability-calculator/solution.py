import torch

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring, rounded to 5 decimal places
    """
    if k < 0:
        return 0.0
    if lam <= 0:
        return 0.0

    k_tensor = torch.tensor(k, dtype=torch.float32)
    lam_tensor = torch.tensor(lam, dtype=torch.float32)

    # Use log-space calculation for numerical stability
    # log(P) = k * log(lam) - lam - log(k!)
    # log(k!) is equivalent to lgamma(k + 1)
    log_prob = (k_tensor * torch.log(lam_tensor)) - lam_tensor - torch.lgamma(k_tensor + 1.0)
    
    # Convert back to linear space
    prob = torch.exp(log_prob)
    
    return round(float(prob.item()), 5)