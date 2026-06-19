import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # Iterate through each feature (column)
    for col_idx in range(X_clean.shape[1]):
        col = X_clean[:, col_idx]
        
        # Check if the column has any NaN values
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            # Compute the median of the non-NaN values in this column
            col_median = np.nanmedian(col)
            
            # Replace NaNs with the computed median
            X_clean[nan_mask, col_idx] = col_median
            
    return X_clean