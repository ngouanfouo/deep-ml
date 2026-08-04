import torch

def law_of_total_probability(priors: torch.Tensor, conditionals: torch.Tensor) -> float:
    """
    Compute P(A) using the Law of Total Probability.
    
    Args:
        priors: 1D tensor containing P(Bi) for each partition event
        conditionals: 1D tensor containing P(A|Bi) for each partition event
    
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    # Using dot product for efficiency
    total_prob = torch.dot(priors, conditionals)
    
    # Round to 4 decimal places and convert to float
    return float(torch.round(total_prob * 10000) / 10000)