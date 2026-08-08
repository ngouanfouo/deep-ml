import math

def matthews_correlation_coefficient(y_true, y_pred) -> float:
    """
    Calculate the Matthews Correlation Coefficient (MCC) for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value as a float, rounded to 4 decimal places
        Returns 0.0 if denominator is zero
    """
    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    # Initialize confusion matrix components
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    # Calculate confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    
    # MCC formula: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    numerator = tp * tn - fp * fn
    
    # Check denominator for zero (when any row or column is all zeros)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    mcc = numerator / denominator
    
    # Round to 4 decimal places
    return round(mcc, 4)