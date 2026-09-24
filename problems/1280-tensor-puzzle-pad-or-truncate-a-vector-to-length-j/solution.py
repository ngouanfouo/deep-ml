import torch

def pad_to(a: torch.Tensor, j: int) -> torch.Tensor:
    """Pad with zeros or truncate a so its length is exactly j."""
    out = torch.zeros(j, dtype=torch.float64, device=a.device)
    n = min(a.shape[0], j)
    if n > 0:
        out[:n] = a[:n].to(torch.float64)
    return out