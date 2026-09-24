import torch

def scatter_add(values: torch.Tensor, indices: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Sum values into bins given by indices (no Tensor.scatter_add)."""
    out = torch.zeros(n_bins, dtype=values.dtype, device=values.device)
    out.index_add_(0, indices, values)
    return out