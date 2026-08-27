import numpy as np

def vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack two 1-D arrays as rows of a (2, n) matrix."""
    # Create mask: [1, 0] reshaped to (2, 1)
    mask = np.array([1.0, 0.0])[:, None]
    
    # mask * a + (1 - mask) * b
    # Selects a for row 0 and b for row 1
    return mask * a + (1 - mask) * b