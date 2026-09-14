import torch
import torch.nn.functional as F

def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mish activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Mish activation value rounded to 4 decimal places
    """
    # Mish formula: x * tanh(softplus(x))
    output = x * torch.tanh(F.softplus(x))
    
    # Round to 4 decimal places
    return torch.round(output, decimals=4)