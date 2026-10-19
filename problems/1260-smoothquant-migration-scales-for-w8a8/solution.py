import numpy as np

def smoothquant(X: np.ndarray, W: np.ndarray, alpha: float = 0.5):
    """
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    X_smooth = X/s, W_smooth = W*s
    If denom 0 use 1e-8; if numerator 0 set s_j=1e-8.
    X: (T, in), W: (out, in)
    Returns s, X_smooth, W_smooth
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    max_abs_X = np.max(np.abs(X), axis=0)  # (in,)
    max_abs_W = np.max(np.abs(W), axis=0)  # (in,)

    denom = np.maximum(max_abs_W, 1e-8)

    s = np.where(
        max_abs_X == 0.0,
        1e-8,
        (max_abs_X ** alpha) / (denom ** (1.0 - alpha)),
    )

    X_smooth = X / s
    W_smooth = W * s

    return s, X_smooth, W_smooth