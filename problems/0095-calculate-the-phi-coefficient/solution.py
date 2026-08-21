import torch

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables using PyTorch.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Calculate contingency table values
    n11 = torch.sum((x_tensor == 1) & (y_tensor == 1)).float()
    n10 = torch.sum((x_tensor == 1) & (y_tensor == 0)).float()
    n01 = torch.sum((x_tensor == 0) & (y_tensor == 1)).float()
    n00 = torch.sum((x_tensor == 0) & (y_tensor == 0)).float()
    
    n = len(x)
    
    # Calculate Phi coefficient
    numerator = (n11 * n00) - (n10 * n01)
    denominator = torch.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    
    # Handle division by zero
    if denominator == 0:
        phi = torch.tensor(0.0)
    else:
        phi = numerator / denominator
    
    # Round to 4 decimal places and return as float
    return round(phi.item(), 4)