import torch

def bucketize(v: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Bucket index = how many boundaries each value is >= (no torch.bucketize)."""
    # v: (N,), boundaries: (M,)
    # Broadcast: (N, 1) >= (1, M) -> (N, M) boolean
    cmp = v.unsqueeze(1) >= boundaries.unsqueeze(0)
    # Count how many boundaries each value is >=
    return cmp.sum(dim=1)