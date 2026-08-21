import torch
import torch.nn.functional as F

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function using PyTorch.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value rounded to 4 decimal places
    """
    # Convert input to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Apply ELU using PyTorch's built-in function
    result = F.elu(x_tensor, alpha=alpha)
    
    # Round to 4 decimal places and return as float
    return round(result.item(), 4)