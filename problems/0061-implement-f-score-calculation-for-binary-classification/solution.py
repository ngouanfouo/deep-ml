import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    # Compute confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision and recall (avoid division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Compute F-beta score
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        f = 0.0
    else:
        f = (1 + beta_sq) * (precision * recall) / denominator
    
    # Round to 3 decimal places
    return round(f, 3)