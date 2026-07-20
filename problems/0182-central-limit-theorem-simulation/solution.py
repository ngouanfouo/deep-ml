import torch

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem using PyTorch tensors.
    """
    torch.manual_seed(seed)
    
    # 1. Determine distribution parameters and draw samples
    # Shape of samples matrix: (runs, n)
    if distribution.lower() == 'uniform':
        true_mean = 0.5
        true_std = 1.0 / (12.0 ** 0.5)
        samples = torch.rand(runs, n)
        
    elif distribution.lower() == 'exponential':
        true_mean = 1.0
        true_std = 1.0
        # PyTorch exponential takes rate (1/scale). scale=1.0 -> rate=1.0
        samples = torch.empty(runs, n).exponential_(1.0)
        
    elif distribution.lower() == 'bernoulli':
        p = 0.3
        true_mean = p
        true_std = (p * (1.0 - p)) ** 0.5
        samples = torch.bernoulli(torch.full((runs, n), p))
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
        
    # 2. Compute the sample mean for each run
    sample_means = torch.mean(samples, dim=1)
    
    # 3. Standardize the sample means to Z-scores
    # Z = (Sample Mean - True Mean) / Standard Error
    standard_error = true_std / (n ** 0.5)
    z_scores = (sample_means - true_mean) / standard_error
    
    # 4. Calculate final metrics
    metrics = {
        'mean': round(torch.mean(z_scores).item(), 3),
        'std': round(torch.std(z_scores, unbiased=True).item(), 3)
    }
    
    return metrics