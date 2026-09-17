import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    X = np.asarray(M, dtype=float)

    # Normalize by Frobenius norm so the largest singular value <= 1.
    norm = np.linalg.norm(X, 'fro')
    if norm == 0.0:
        return X.tolist()
    X = X / norm

    # Iterate: X <- a*X + b*(X X^T) X + c*(X X^T)^2 X
    # Reuse intermediates to avoid redundant matrix products:
    #   A   = X X^T
    #   AX  = A X
    #   AAX = A (A X)
    for _ in range(num_iters):
        A = X @ X.T
        AX = A @ X
        AAX = A @ AX
        X = a * X + b * AX + c * AAX

    return X.tolist()