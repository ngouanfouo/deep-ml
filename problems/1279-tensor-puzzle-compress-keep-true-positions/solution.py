import torch

def compress(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Keep values where g is True, packed left; pad right with zeros."""
    selected = v[g]
    out = torch.zeros_like(v)
    out[:selected.shape[0]] = selected
    return out