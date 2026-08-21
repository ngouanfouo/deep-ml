import torch

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between two tensors.

    Parameters:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Convert to float for calculations
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Calculate absolute differences
    abs_diff = torch.abs(y_true - y_pred)
    
    # Calculate mean
    mae_value = torch.mean(abs_diff)
    
    # Return as float
    return mae_value.item()