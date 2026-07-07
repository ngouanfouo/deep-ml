import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Return the (feature_index, threshold) that minimises weighted Gini impurity.
    A split sends samples where X[:, feature] <= threshold to the left child,
    and the remaining samples to the right child.
    """
    n_samples, n_features = X.shape
    
    best_gini = float('inf')
    best_feature_idx = -1
    best_threshold = -1.0
    
    # Iterate through all available features
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        # Get unique feature values to use as candidate thresholds
        thresholds = np.unique(feature_values)
        
        for threshold in thresholds:
            # Create boolean masks for left and right branches
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            n_left = len(y_left)
            n_right = len(y_right)
            
            # Skip invalid splits that put all data into one leaf
            if n_left == 0 or n_right == 0:
                continue
                
            # Compute Gini Impurity for Left Child
            p_left_1 = np.sum(y_left == 1) / n_left
            p_left_0 = 1.0 - p_left_1
            gini_left = 1.0 - (p_left_1**2 + p_left_0**2)
            
            # Compute Gini Impurity for Right Child
            p_right_1 = np.sum(y_right == 1) / n_right
            p_right_0 = 1.0 - p_right_1
            gini_right = 1.0 - (p_right_1**2 + p_right_0**2)
            
            # Compute Weighted Gini Impurity for the total split
            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            
            # Keep the first split that strictly improves the metric (handles ties gracefully)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_idx = feature_idx
                best_threshold = threshold
                
    return best_feature_idx, best_threshold