import numpy as np

def temporal_aggregate(features, k: int):
    """
    Aggregate frame features by averaging every k consecutive frames.

    Args:
        features: 2D array-like of shape (num_frames, feature_dim)
        k: aggregation factor (positive int)

    Returns:
        Nested list of shape (num_frames // k, feature_dim)
    """
    features = np.array(features, dtype=float)
    num_frames = features.shape[0]
    
    # Number of complete groups
    num_groups = num_frames // k
    
    if num_groups == 0:
        # Not enough frames for even one group
        return []
    
    # Trim to a multiple of k, then reshape and average
    trimmed = features[:num_groups * k]
    grouped = trimmed.reshape(num_groups, k, features.shape[1])
    aggregated = grouped.mean(axis=1)
    
    return aggregated.tolist()