import torch

def repeat_rows(a: torch.Tensor, d: int) -> torch.Tensor:
    """Repeat vector a as d identical rows (no torch.repeat/tile)."""
    ones = torch.ones((d, 1), dtype=a.dtype, device=a.device)
    return ones * a