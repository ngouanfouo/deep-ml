import math
import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                    in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    # Convert to a tensor if it isn't already, to support both list and tensor inputs
    if not isinstance(probabilities, torch.Tensor):
        probs = torch.tensor(probabilities, dtype=torch.float32)
    else:
        probs = probabilities
        
    # Calculate the negative average log probability (cross-entropy)
    # Using natural log (math.log / torch.log)
    n = len(probs)
    log_probs = torch.log(probs)
    mean_negative_log_prob = -torch.sum(log_probs) / n
    
    # Calculate perplexity using the exponential function
    perplexity = torch.exp(mean_negative_log_prob)
    
    return float(perplexity.item())