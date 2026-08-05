import numpy as np

def precision(y_true, y_pred):
    # Compute true positives and false positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # Avoid division by zero
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)