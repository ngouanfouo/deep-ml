import torch

def birthday_problem(n: int, days: int = 365) -> float:
    """
    Calculate the probability that at least two people share the same birthday.
    
    Args:
        n: Number of people in the group
        days: Number of days in a year (default 365)
    
    Returns:
        float: Probability of at least one shared birthday, rounded to 4 decimal places
    """
    # Edge cases
    if n <= 1:
        return 0.0
    
    if n > days:
        return 1.0
    
    # Using PyTorch for computation
    days_tensor = torch.tensor(days, dtype=torch.float64)
    n_tensor = torch.tensor(n, dtype=torch.float64)
    
    # Compute log probabilities for numerical stability
    # log(P(no match)) = sum_{i=0}^{n-1} log((days - i) / days)
    i_values = torch.arange(n, dtype=torch.float64)
    log_probs = torch.log((days_tensor - i_values) / days_tensor)
    log_prob_no_match = torch.sum(log_probs)
    
    # Convert back to probability
    prob_no_match = torch.exp(log_prob_no_match).item()
    
    # Probability of at least one match
    prob_match = 1.0 - prob_no_match
    
    # Round to 4 decimal places
    return round(prob_match, 4)