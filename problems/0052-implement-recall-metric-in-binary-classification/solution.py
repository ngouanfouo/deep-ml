import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # True Positives: predicted 1 and actual 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Negatives: predicted 0 but actual 1
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Avoid division by zero
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)