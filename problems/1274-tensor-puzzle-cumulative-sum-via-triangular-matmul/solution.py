import torch

def cumsum(a: torch.Tensor) -> torch.Tensor:
    """Prefix sums via lower-triangular matmul (no torch.cumsum)."""
    n = len(a)
    # Create lower-triangular matrix of ones where i >= j
    L = (torch.arange(n)[:, None] >= torch.arange(n)[None, :]).float()
    # Matrix-vector multiplication gives cumulative sums
    return L @ a