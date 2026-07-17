from collections import Counter

def confusion_matrix(data):
    """
    Generates a 2x2 confusion matrix from binary classification pairs.
    
    Structure:
    [[Actual 1 & Pred 1 (TP), Actual 1 & Pred 0 (FN)],
     [Actual 0 & Pred 1 (FP), Actual 0 & Pred 0 (TN)]]
    """
    # Count occurrences of each unique (y_true, y_pred) pair
    counts = Counter((y_true, y_pred) for y_true, y_pred in data)
    
    # Extract coordinates based on the (Actual, Predicted) keys
    actual_1_pred_1 = counts[(1, 1)]
    actual_1_pred_0 = counts[(1, 0)]
    actual_0_pred_1 = counts[(0, 1)]
    actual_0_pred_0 = counts[(0, 0)]
    
    # Return the matrix matching the exact expected configurations
    return [
        [actual_1_pred_1, actual_1_pred_0],
        [actual_0_pred_1, actual_0_pred_0]
    ]