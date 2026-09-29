import numpy as np

def weighted_index_score(Q, K, W):
    """
    Compute weighted multi-head index scores.

    Args:
        Q: array-like of shape (T, H, D)
        K: array-like of shape (S, D)
        W: array-like of shape (T, H)

    Returns:
        Nested list of shape (T, S) with index scores.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    W = np.asarray(W, dtype=float)

    # Q: (T, H, D), K: (S, D) -> dots: (T, H, S)
    dots = Q @ K.T

    # Apply ReLU
    activated = np.maximum(dots, 0.0)

    # Weight each head: W[:, :, None] has shape (T, H, 1)
    weighted = W[:, :, None] * activated  # (T, H, S)

    # Sum over heads -> (T, S)
    scores = weighted.sum(axis=1)

    return scores.tolist()