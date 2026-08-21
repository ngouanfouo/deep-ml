import torch

def random_forest_feature_importance(trees: list, n_features: int) -> torch.Tensor:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        torch.Tensor of feature importances normalized to sum to 1.0
    """
    # Check for empty forest
    if not trees:
        return torch.zeros(n_features, dtype=torch.float32)
    
    # Initialize importance scores
    importance_scores = torch.zeros(n_features, dtype=torch.float32)
    
    # Aggregate impurity decreases across all trees
    for tree in trees:
        for split in tree:
            feature_idx = split['feature_index']
            impurity_decrease = split['impurity_decrease']
            importance_scores[feature_idx] += impurity_decrease
    
    # Normalize to sum to 1.0
    total_importance = torch.sum(importance_scores)
    if total_importance > 0:
        importance_scores = importance_scores / total_importance
    
    return importance_scores