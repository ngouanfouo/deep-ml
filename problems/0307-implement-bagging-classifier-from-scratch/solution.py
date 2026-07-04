import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_test = X_test.shape[0]
    
    # Store all stump predictions for test data
    test_predictions = np.zeros((n_estimators, n_test))
    
    for estimator_idx in range(n_estimators):
        # Bootstrap sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Train a decision stump on the bootstrap sample
        best_feature = 0
        best_threshold = 0
        best_direction = 1  # 1 for <= threshold, -1 for > threshold
        best_error = float('inf')
        
        # For each feature
        for feature_idx in range(n_features):
            feature_values = X_bootstrap[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try each unique value as a threshold
            for threshold in unique_values:
                # Try both directions
                for direction in [1, -1]:
                    # Predict based on direction and threshold
                    if direction == 1:
                        # Predict class 1 if feature <= threshold
                        preds = np.where(feature_values <= threshold, 1, 0)
                    else:
                        # Predict class 1 if feature > threshold
                        preds = np.where(feature_values > threshold, 1, 0)
                    
                    # Calculate error
                    error = np.mean(preds != y_bootstrap)
                    
                    if error < best_error:
                        best_error = error
                        best_feature = feature_idx
                        best_threshold = threshold
                        best_direction = direction
        
        # Make predictions on test data using the best stump
        if best_direction == 1:
            # Predict class 1 if feature <= threshold
            test_preds = np.where(X_test[:, best_feature] <= best_threshold, 1, 0)
        else:
            # Predict class 1 if feature > threshold
            test_preds = np.where(X_test[:, best_feature] > best_threshold, 1, 0)
        
        test_predictions[estimator_idx] = test_preds
    
    # Combine predictions via majority voting
    final_predictions = np.mean(test_predictions, axis=0) >= 0.5
    return final_predictions.astype(int)