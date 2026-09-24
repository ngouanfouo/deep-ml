import torch

def roll(a: torch.Tensor) -> torch.Tensor:
    """Circular left-shift by one via index arithmetic (no torch.roll)."""
    n = a.shape[0]
    if n == 0:
        return a
    idx = torch.arange(n, device=a.device)
    shifted_idx = (idx + 1) % n
    return a[shifted_idx]