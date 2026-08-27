import numpy as np

def outer(a: np.ndarray, b: np.ndarray):
    """Compute outer product of 1-D arrays a and b using broadcasting."""
    # a[:, None] makes a a column vector, b[None, :] makes b a row vector
    return a[:, None] * b[None, :]