import torch

def entropy_split_selection(X: torch.Tensor, y: torch.Tensor) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    n_samples, n_features = X.shape
    
    # Handle edge case: empty dataset
    if n_samples == 0:
        return (0, 0.0, 0.0)
    
    # Handle edge case: pure node (all samples have the same label)
    unique_labels = torch.unique(y)
    if len(unique_labels) == 1:
        # Find any valid threshold to return
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = torch.unique(feature_values)
            if len(unique_values) > 1:
                # Return midpoint of the first two unique values
                threshold = (unique_values[0] + unique_values[1]) / 2.0
                return (feature_idx, threshold.item(), 0.0)
        # If all features have only one unique value, return 0.0
        return (0, 0.0, 0.0)
    
    # Compute parent entropy
    parent_entropy = compute_entropy(y)
    
    best_info_gain = -1.0
    best_feature_index = 0
    best_threshold = 0.0
    
    # For each feature, find the best split
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Get sorted unique values
        unique_values = torch.unique(feature_values)
        
        # Skip if feature has only one unique value
        if len(unique_values) <= 1:
            continue
        
        # Consider midpoints between consecutive unique values as thresholds
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        
        # Evaluate each threshold
        for threshold in thresholds:
            # Split the data
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            # Skip if either split is empty
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            
            # Get labels for left and right splits
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # Compute entropies
            entropy_left = compute_entropy(y_left)
            entropy_right = compute_entropy(y_right)
            
            # Compute weighted average entropy
            weight_left = len(y_left) / n_samples
            weight_right = len(y_right) / n_samples
            weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
            
            # Compute information gain
            info_gain = parent_entropy - weighted_entropy
            
            # Update best split if this one is better
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = feature_idx
                best_threshold = threshold.item()
    
    # If no valid split found (e.g., all features have single unique values)
    if best_info_gain == -1.0:
        return (0, 0.0, 0.0)
    
    return (best_feature_index, best_threshold, best_info_gain)


def compute_entropy(y: torch.Tensor) -> float:
    """
    Compute entropy of labels.
    
    Entropy = -sum(p_i * log2(p_i)) where p_i is the proportion of class i.
    """
    if len(y) == 0:
        return 0.0
    
    # Get unique labels and their counts
    unique_labels, counts = torch.unique(y, return_counts=True)
    
    # Compute probabilities
    probs = counts.float() / len(y)
    
    # Compute entropy: -sum(p * log2(p))
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log2(probs + eps))
    
    return entropy.item()