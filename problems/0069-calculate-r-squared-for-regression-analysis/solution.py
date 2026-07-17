import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) value.
    
    :param y_true: np.ndarray, the true target values
    :param y_pred: np.ndarray, the predicted target values
    :return: float, the R-squared value rounded to 3 decimal places
    """
    # Calculate the mean of the true values
    y_true_mean = np.mean(y_true)
    
    # Residual sum of squares (unexplained variance)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total sum of squares (total variance in data)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    
    # Prevent division by zero if all y_true values are identical
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
        
    r2_value = 1.0 - (ss_res / ss_tot)
    
    return round(float(r2_value), 3)