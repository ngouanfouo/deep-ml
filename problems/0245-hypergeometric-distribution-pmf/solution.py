import torch

def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    # Check if k is outside the valid range
    lower_bound = max(0, n - (N - K))
    upper_bound = min(n, K)
    
    if k < lower_bound or k > upper_bound:
        return 0.0
    
    # Convert to tensors for computation
    N_t = torch.tensor(N, dtype=torch.float64)
    K_t = torch.tensor(K, dtype=torch.float64)
    n_t = torch.tensor(n, dtype=torch.float64)
    k_t = torch.tensor(k, dtype=torch.float64)
    
    # Calculate log probabilities for numerical stability
    # log(C(a,b)) = log(a!) - log(b!) - log((a-b)!)
    def log_comb(a, b):
        if b < 0 or b > a:
            return float('-inf')
        return torch.lgamma(a + 1) - torch.lgamma(b + 1) - torch.lgamma(a - b + 1)
    
    log_prob = log_comb(K_t, k_t) + log_comb(N_t - K_t, n_t - k_t) - log_comb(N_t, n_t)
    
    # Convert back to probability
    probability = torch.exp(log_prob).item()
    
    # Round to 4 decimal places
    return round(probability, 4)