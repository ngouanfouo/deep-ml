import torch

def incremental_mean(Q_prev: torch.Tensor, k: int, R: torch.Tensor) -> torch.Tensor:
    """
    Q_prev: previous mean estimate (torch.Tensor)
    k: number of times the action has been selected (int)
    R: new observed reward (torch.Tensor)
    Returns: new mean estimate (torch.Tensor)
    """
    Q_prev = torch.as_tensor(Q_prev, dtype=torch.float)
    R = torch.as_tensor(R, dtype=torch.float)
    return Q_prev + (R - Q_prev) / k