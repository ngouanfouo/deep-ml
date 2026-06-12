import numpy as np

def divide_on_feature(X, feature_i, threshold):
    """
    Divide a dataset into two subsets based on whether the value of a 
    specified feature is greater than or equal to a given threshold.

    Args:
        X: 2D numpy array of shape (n_samples, n_features)
        feature_i: Int, the index of the feature column to split on
        threshold: Float/Int, the split threshold value

    Returns:
        A list containing two 2D numpy arrays:
        [X_meet_condition, X_not_meet_condition]
    """
    # Create a boolean condition mask for rows where the specified feature column >= threshold
    condition = X[:, feature_i] >= threshold
    
    # Use the boolean mask to extract rows that meet the condition
    X_meet = X[condition]
    
    # Use the bitwise NOT operator (~) to extract rows that do not meet the condition
    X_not_meet = X[~condition]
    
    return [X_meet, X_not_meet]

