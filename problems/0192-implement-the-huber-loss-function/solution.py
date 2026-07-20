import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    # 1. Standardize inputs into NumPy arrays
    true_arr = np.atleast_1d(np.array(y_true, dtype=np.float64))
    pred_arr = np.atleast_1d(np.array(y_pred, dtype=np.float64))
    
    # 2. Compute absolute residuals
    error = true_arr - pred_arr
    abs_error = np.abs(error)
    
    # 3. Apply the conditional piecewise logic using vector operations
    mse_mask = abs_error <= delta
    
    loss = np.zeros_like(abs_error)
    
    # Quadratic behavior for small errors
    loss[mse_mask] = 0.5 * (error[mse_mask] ** 2)
    
    # Linear behavior for large outliers
    loss[~mse_mask] = delta * (abs_error[~mse_mask] - 0.5 * delta)
    
    # 4. Return the statistical mean scalar
    return float(np.mean(loss))