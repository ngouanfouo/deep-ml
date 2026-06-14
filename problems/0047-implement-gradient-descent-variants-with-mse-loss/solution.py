import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m = X.shape[0]  # number of samples
    
    for epoch in range(n_epochs):
        if method == 'batch':
            # Batch GD: use all samples for one gradient update per epoch
            predictions = X @ weights
            errors = predictions - y
            gradient = (2 / m) * (X.T @ errors)
            weights = weights - learning_rate * gradient
            
        elif method == 'stochastic':
            # Stochastic GD: update weights after each sample
            for i in range(m):
                prediction = X[i] @ weights
                error = prediction - y[i]
                gradient = 2 * error * X[i]
                weights = weights - learning_rate * gradient
                
        elif method == 'mini_batch':
            # Mini-batch GD: process batches of consecutive samples
            for i in range(0, m, batch_size):
                # Get batch (without wrapping at the end)
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                batch_m = X_batch.shape[0]
                
                predictions = X_batch @ weights
                errors = predictions - y_batch
                gradient = (2 / batch_m) * (X_batch.T @ errors)
                weights = weights - learning_rate * gradient
    
    return weights