import torch

def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Dice Score (Sørensen-Dice coefficient) for binary classification.

    Args:
        y_true: Binary tensor of true labels.
        y_pred: Binary tensor of predicted labels.

    Returns:
        Dice Score as a float rounded to 3 decimal places.
    """
    # Force conversion to float to handle numerical updates securely
    y_true_f = y_true.float()
    y_pred_f = y_pred.float()
    
    # Calculate intersection and total absolute sums
    intersection = torch.sum(y_true_f * y_pred_f).item()
    total_elements = (torch.sum(y_true_f) + torch.sum(y_pred_f)).item()
    
    # Edge Case Handling: If there are no positive predictions or true positives, 
    # return 0.0 to prevent division by zero and match the expected test standard.
    if total_elements == 0:
        return 0.0
        
    res = (2.0 * intersection) / total_elements
    
    return round(float(res), 3)