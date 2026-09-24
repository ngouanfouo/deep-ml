import torch

def heaviside(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Heaviside step: 0 if a<0, b if a==0, 1 if a>0 (no torch.heaviside)."""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.where(a < 0, zeros, torch.where(a == 0, b, ones))