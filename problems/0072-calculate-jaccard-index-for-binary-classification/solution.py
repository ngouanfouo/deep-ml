import numpy as np

def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard Index (Intersection over Union) for binary sets.
    
    Args:
        y_true: np.ndarray of true binary labels (0 or 1)
        y_pred: np.ndarray of predicted binary labels (0 or 1)
        
    Returns:
        float: Jaccard index value rounded to 3 decimal places.
    """
    # Ensure inputs are evaluated as boolean masks/binary targets
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    
    # Calculate intersection and union sizes
    intersection = np.sum(y_true_bool & y_pred_bool)
    union = np.sum(y_true_bool | y_pred_bool)
    
    # Edge case handling: If both arrays contain only zeros, the union is 0.
    # By convention, if there are no positives anywhere, the overlap is perfect (1.0).
    if union == 0:
        return 1.0
        
    result = float(intersection) / float(union)
    
    return round(result, 3)