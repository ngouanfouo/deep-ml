import torch

def diff(a: torch.Tensor) -> torch.Tensor:
    """Return length-n array with first element and successive differences."""
    # For i >= 1, compute a[i] - a[i-1]
    differences = a[1:] - a[:-1]
    
    # Prepend the first element
    result = torch.cat([a[:1], differences])
    
    return result