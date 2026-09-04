import numpy as np

def batch_size_scaled_sgd_loss(X: np.ndarray, y: np.ndarray, w0: np.ndarray,
                               base_lr: float, base_bs: int,
                               batch_size: int, epochs: int) -> float:
    """
    Run mini-batch SGD for linear regression with the linear LR scaling rule
    (effective_lr = base_lr * batch_size / base_bs) and return the final MSE loss.
    """
    n, d = X.shape
    w = w0.copy()
    
    # Apply linear scaling rule to adjust learning rate
    effective_lr = base_lr * (batch_size / base_bs)
    
    for epoch in range(epochs):
        # Process samples in their given order (no shuffling)
        for start_idx in range(0, n, batch_size):
            # Get the current mini-batch (last batch may be smaller)
            end_idx = min(start_idx + batch_size, n)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Compute predictions for the batch
            y_pred = X_batch @ w
            
            # Compute gradient: mean squared error gradient
            # dL/dw = (2/n_batch) * X_batch.T @ (y_pred - y_batch)
            n_batch = end_idx - start_idx
            gradient = (2.0 / n_batch) * X_batch.T @ (y_pred - y_batch)
            
            # Update weights using the effective learning rate
            w -= effective_lr * gradient
    
    # Calculate final MSE loss on the full dataset
    y_pred_full = X @ w
    mse = np.mean((y_pred_full - y) ** 2)
    
    return float(mse)