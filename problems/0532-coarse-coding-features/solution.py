import numpy as np

def coarse_coding(state: tuple, receptive_fields: list, weights: np.ndarray = None) -> tuple:
    """
    Compute binary coarse coding features for a 2D continuous state.
    
    Args:
        state: tuple (x, y) representing a point in 2D continuous space
        receptive_fields: list of tuples (center_x, center_y, radius) defining circular regions
        weights: optional numpy array of shape (n,) for linear value estimation
    
    Returns:
        tuple: (features, value)
            - features: numpy array of 0s and 1s with shape (n,)
            - value: float if weights provided, else None
    """
    x, y = state

    features = []
    for cx, cy, radius in receptive_fields:
        # Euclidean distance from the state to this field's center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # Feature is active if the state lies within (or on) the boundary
        features.append(1 if dist <= radius else 0)

    features = np.array(features, dtype=int)

    if weights is not None:
        value = float(np.dot(features, weights))
    else:
        value = None

    return features, value