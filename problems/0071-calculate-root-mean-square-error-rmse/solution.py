import torch
import torch.nn.functional as F

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        RMSE value rounded to three decimal places.
        Returns ValueError or TypeError for invalid edge cases.
    """
    # 1. Check for invalid input types
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("Inputs must be PyTorch Tensors.")
        
    # 2. Check for mismatched array shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
        
    # 3. Check for empty arrays
    if y_true.numel() == 0:
        raise ValueError("Cannot calculate RMSE on empty tensors.")
        
    # Cast tensors to float32 or float64 to ensure accurate decimal computation
    y_true_f = y_true.to(torch.float32)
    y_pred_f = y_pred.to(torch.float32)
    
    # 4. Calculate Mean Squared Error (MSE) first
    mse = F.mse_loss(y_pred_f, y_true_f, reduction='mean')
    
    # 5. Take the square root to find RMSE
    rmse_val = torch.sqrt(mse).item()
    
    return round(float(rmse_val), 3)