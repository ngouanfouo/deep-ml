import torch

def calculate_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the F1 score based on true and predicted labels using PyTorch.

    Args:
        y_true (torch.Tensor): True labels (ground truth).
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    # Convert to float for calculations
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Calculate true positives, false positives, and false negatives
    tp = torch.sum((y_pred == 1) & (y_true == 1)).float()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).float()
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # If f1 is a tensor, convert to float; otherwise use as is
    if torch.is_tensor(f1):
        f1 = f1.item()
    
    # Round to 3 decimal places and return as float
    return round(f1, 3)