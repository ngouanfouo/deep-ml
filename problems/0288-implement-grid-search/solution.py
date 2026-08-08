import torch
from itertools import product
import warnings

def grid_search(X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor,
                param_grid: dict, model_fn: callable,
                scoring_fn: callable) -> tuple:
    """
    Perform grid search to find optimal hyperparameters.

    Args:
        X_train: Training features as torch.Tensor
        y_train: Training labels as torch.Tensor
        X_val: Validation features as torch.Tensor
        y_val: Validation labels as torch.Tensor
        param_grid: Dict mapping parameter names to lists of values to try
        model_fn: Function(X_train, y_train, X_val, **params) -> predictions (torch.Tensor)
        scoring_fn: Function(y_true, y_pred) -> score (higher is better)

    Returns:
        Tuple of (best_params dict, best_score rounded to 4 decimals)

    Example:
        >>> def knn_model(X_train, y_train, X_val, k=1):
        ...     # Simple k-NN implementation for demonstration
        ...     predictions = []
        ...     for x in X_val:
        ...         distances = torch.cdist(x.unsqueeze(0), X_train)
        ...         _, indices = torch.topk(distances, k, dim=1, largest=False)
        ...         k_labels = y_train[indices.squeeze()]
        ...         pred = torch.mode(k_labels).values
        ...         predictions.append(pred)
        ...     return torch.tensor(predictions)
        >>> 
        >>> def accuracy(y_true, y_pred):
        ...     return (y_true == y_pred).float().mean().item()
        >>> 
        >>> X_train = torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.float32)
        >>> y_train = torch.tensor([0, 1, 1, 0, 0])
        >>> X_val = torch.tensor([[1.5, 0]], dtype=torch.float32)
        >>> y_val = torch.tensor([1])
        >>> param_grid = {'k': [1, 3, 5]}
        >>> best_params, best_score = grid_search(X_train, y_train, X_val, y_val, 
        ...                                     param_grid, knn_model, accuracy)
        >>> print(best_params, best_score)
        {'k': 1} 1.0
    """
    # Input validation
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid must be a dictionary")
    
    if not param_grid:
        raise ValueError("param_grid cannot be empty")
    
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have same number of samples")
    
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have same number of samples")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Check that all values are lists
    for name, values in zip(param_names, param_values):
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"Values for parameter '{name}' must be a list or tuple")
        if len(values) == 0:
            raise ValueError(f"Parameter '{name}' has empty value list")
    
    # Calculate total number of combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Searching {total_combinations} parameter combinations...")
    
    best_score = -float('inf')
    best_params = None
    all_results = []  # Store all results for debugging
    
    # Evaluate each combination
    for idx, combination in enumerate(product(*param_values)):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))
        
        # Print progress
        print(f"  Evaluating {idx+1}/{total_combinations}: {params}")
        
        try:
            # Train and predict
            predictions = model_fn(X_train, y_train, X_val, **params)
            
            # Ensure predictions are torch.Tensor
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            
            # Score
            score = scoring_fn(y_val, predictions)
            
            # Store result
            all_results.append((params, score))
            
            # Update best if this score is better
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"    New best score: {best_score:.4f}")
                
        except Exception as e:
            warnings.warn(f"Failed for params {params}: {e}")
            all_results.append((params, None))
            continue
    
    # If no valid combination was found
    if best_params is None:
        raise RuntimeError("No valid parameter combination found. Check your model and scoring functions.")
    
    # Round the best score
    best_score_rounded = round(float(best_score), 4)
    
    # Print summary
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score_rounded}")
    
    return best_params, best_score_rounded