import torch

def stratified_train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float,
    random_seed: int = None
):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix tensor of shape (n_samples, n_features)
        y: Label tensor of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test as torch.Tensors
    """
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Input validation
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. X: {X.shape[0]}, y: {y.shape[0]}")
    
    # Get unique classes
    unique_classes = torch.unique(y)
    n_samples = len(y)
    
    # Initialize lists to collect train and test indices
    train_indices = []
    test_indices = []
    
    # Process each class separately
    for cls in unique_classes:
        # Get indices of samples belonging to this class
        class_indices = torch.where(y == cls)[0].tolist()
        n_class_samples = len(class_indices)
        
        # Calculate number of samples for test set for this class
        # Use int() for truncation (matches numpy's behavior)
        n_test = int(n_class_samples * test_size)
        n_train = n_class_samples - n_test
        
        # Ensure at least one sample in test set if possible
        if n_test == 0 and n_class_samples > 0:
            n_test = 1
            n_train = n_class_samples - 1
        
        # Shuffle indices within the class
        shuffled_indices = torch.randperm(n_class_samples).tolist()
        shuffled_class_indices = [class_indices[i] for i in shuffled_indices]
        
        # Split the shuffled indices
        test_indices.extend(shuffled_class_indices[:n_test])
        train_indices.extend(shuffled_class_indices[n_test:])
    
    # Convert indices to tensors
    train_indices = torch.tensor(train_indices, dtype=torch.long)
    test_indices = torch.tensor(test_indices, dtype=torch.long)
    
    # Shuffle the final train and test indices for randomness
    if len(train_indices) > 0:
        train_indices = train_indices[torch.randperm(len(train_indices))]
    if len(test_indices) > 0:
        test_indices = test_indices[torch.randperm(len(test_indices))]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test