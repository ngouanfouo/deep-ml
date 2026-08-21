import torch

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1) -> torch.Tensor:
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists or Tensor), shape (n_samples, n_features)
        y: Target values (list or Tensor), shape (n_samples,)
        current_predictions: Current ensemble predictions (list or Tensor), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        torch.Tensor of updated predictions rounded to 4 decimal places
    """
    # Convert inputs to tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(current_predictions, torch.Tensor):
        current_predictions = torch.tensor(current_predictions, dtype=torch.float32)
    
    n_samples, n_features = X.shape
    
    # Step 1: Compute residuals
    residuals = y - current_predictions
    
    # Step 2: Fit decision stump to residuals
    # Initialize best split parameters
    best_mse = float('inf')
    best_left_pred = None
    best_right_pred = None
    best_left_indices = None
    best_right_indices = None
    found_valid_split = False
    
    # Try all features
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Get unique sorted values
        unique_values = torch.unique(feature_values)
        
        # If all values are the same, skip this feature
        if len(unique_values) <= 1:
            continue
        
        # Try all possible split thresholds (midpoints between consecutive unique values)
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
            
            # Split data
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            # Skip if either side is empty
            if torch.sum(left_mask) == 0 or torch.sum(right_mask) == 0:
                continue
            
            # Get residuals for each partition
            left_residuals = residuals[left_mask]
            right_residuals = residuals[right_mask]
            
            # Predict the mean for each partition
            left_pred = torch.mean(left_residuals)
            right_pred = torch.mean(right_residuals)
            
            # Compute MSE for this split
            left_mse = torch.mean((left_residuals - left_pred) ** 2)
            right_mse = torch.mean((right_residuals - right_pred) ** 2)
            
            # Weighted MSE
            total_mse = (len(left_residuals) * left_mse + len(right_residuals) * right_mse) / n_samples
            
            # Update best split if this one is better
            if total_mse < best_mse:
                best_mse = total_mse
                best_left_pred = left_pred
                best_right_pred = right_pred
                best_left_indices = left_mask
                best_right_indices = right_mask
                found_valid_split = True
    
    # Step 3: Make stump predictions
    stump_predictions = torch.zeros(n_samples)
    
    if found_valid_split:
        # Use the best split found
        stump_predictions[best_left_indices] = best_left_pred
        stump_predictions[best_right_indices] = best_right_pred
    else:
        # No valid split found, predict mean of all residuals
        stump_predictions = torch.full((n_samples,), torch.mean(residuals))
    
    # Step 4: Update predictions
    updated_predictions = current_predictions + learning_rate * stump_predictions
    
    # Round to 4 decimal places
    return torch.round(updated_predictions * 10000) / 10000