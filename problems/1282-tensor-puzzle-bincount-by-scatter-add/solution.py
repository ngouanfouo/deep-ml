import torch

def bincount(a: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Count occurrences of each integer 0..n_bins-1 (no torch.bincount)."""
    bins = torch.arange(n_bins, device=a.device)
    # Outer equality: shape (len(a), n_bins)
    eq = (a.unsqueeze(1) == bins.unsqueeze(0))
    # Sum over the sample dimension to get counts per bin
    return eq.sum(dim=0)