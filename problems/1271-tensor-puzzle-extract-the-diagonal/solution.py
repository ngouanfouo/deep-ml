import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    n = A.shape[0]
    # Use np.arange for both row and column indices
    return A[np.arange(n), np.arange(n)]