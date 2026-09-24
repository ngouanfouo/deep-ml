import torch

def linspace(start, stop, n: int) -> torch.Tensor:
    """n evenly spaced values from start to stop inclusive (no torch.linspace)."""
    if n == 1:
        return torch.tensor([start], dtype=torch.float64)
    idx = torch.arange(n, dtype=torch.float64)
    step = (stop - start) / (n - 1)
    return start + idx * step