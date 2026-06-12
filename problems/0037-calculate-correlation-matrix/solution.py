import numpy as np

def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the Pearson correlation matrix for a given dataset.
    
    Args:
        X: 2D numpy array of shape (n_samples, n_features_x)
        Y: Optional 2D numpy array of shape (n_samples, n_features_y)
        
    Returns:
        A 2D numpy array representing the correlation matrix.
    """
    # Ensure inputs are float arrays
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    
    if Y is not None:
        Y = np.asarray(Y, dtype=float)
        if Y.shape[0] != n_samples:
            raise ValueError("X and Y must have the same number of samples (rows).")
    else:
        Y = X

    # 1. Center the matrices by subtracting the mean of each feature (column-wise)
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # 2. Compute the standard deviation of each feature column
    std_X = np.std(X, axis=0)
    std_Y = np.std(Y, axis=0)
    
    # 3. Calculate the Covariance matrix: (X_centered.T @ Y_centered) / n_samples
    covariance_matrix = np.dot(X_centered.T, Y_centered) / n_samples
    
    # 4. Normalize by dividing by the outer product of the standard deviations
    # np.outer(std_X, std_Y) creates a matrix where entry (i, j) is std_X[i] * std_Y[j]
    std_outer = np.outer(std_X, std_Y)
    
    # Handle potential division by zero for features with zero variance
    # replacing 0 with 1 in denominator prevents NaNs (correlation remains 0)
    std_outer[std_outer == 0] = 1.0
    
    correlation_matrix = covariance_matrix / std_outer
    
    return correlation_matrix

