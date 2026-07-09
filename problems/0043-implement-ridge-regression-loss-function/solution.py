import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    """
    Calculate the Ridge Regression loss (MSE + L2 regularization).
    
    Args:
        X: Feature matrix of shape (num_samples, num_features)
        w: Coefficients array of shape (num_features,)
        y_true: True labels array of shape (num_samples,)
        alpha: Regularization strength parameter
        
    Returns:
        The combined Ridge loss value as a float
    """
    # Predict the target values
    y_pred = X @ w
    
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate the L2 regularization penalty (squared norm of weights)
    l2_penalty = alpha * np.sum(w ** 2)
    
    # Total loss is the sum of MSE and the L2 penalty
    total_loss = mse + l2_penalty
    
    return float(total_loss)