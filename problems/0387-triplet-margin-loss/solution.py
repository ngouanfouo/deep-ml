import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    """
    Compute the triplet margin loss for metric learning.
    
    Args:
        anchor: Anchor embeddings, shape (D,) for single or (N, D) for batch
        positive: Positive embeddings (same class as anchor), same shape as anchor
        negative: Negative embeddings (different class from anchor), same shape as anchor
        margin: Minimum desired distance gap between positive and negative pairs
    
    Returns:
        Mean triplet margin loss as a float
    """
    # Ensure inputs are at least 2D to uniformly handle both single samples and batches
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)
    
    # Compute Euclidean distances along the feature dimension (axis=1)
    d_pos = np.linalg.norm(anchor - positive, axis=1)
    d_neg = np.linalg.norm(anchor - negative, axis=1)
    
    # Compute per-triplet loss: max(0, d_pos - d_neg + margin)
    losses = np.maximum(0.0, d_pos - d_neg + margin)
    
    # Return the mean loss over all triplets as a Python float
    return float(np.mean(losses))