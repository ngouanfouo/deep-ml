import torch
import torch.nn.functional as F

def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute temperature-scaled softmax probabilities from logits.
    
    Args:
        logits: 1D torch.Tensor of raw model output scores
        temperature: float controlling distribution sharpness
    
    Returns:
        torch.Tensor of probabilities after temperature scaling
    """
    # Handle temperature <= 0: return one-hot distribution
    if temperature <= 0:
        # Get the index of the maximum value (first occurrence in case of ties)
        max_idx = torch.argmax(logits)
        # Create one-hot distribution
        probs = torch.zeros_like(logits)
        probs[max_idx] = 1.0
        return probs
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax with numerical stability (subtract max)
    # This is handled automatically by F.softmax which is numerically stable
    probs = F.softmax(scaled_logits, dim=0)
    
    return probs