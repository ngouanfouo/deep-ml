import torch

def ones(n: int) -> torch.Tensor:
    """Return a length-n float vector of ones without calling torch.ones."""
    # Create a range [0, 1, 2, ..., n-1], multiply by 0 to get zeros, then add 1
    # Use default dtype (float32) to match expected output format
    result = torch.arange(n) * 0.0 + 1.0
    return result