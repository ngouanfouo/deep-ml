import numpy as np


def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.

    S(w, λ) = sign(w) * max(|w| - λ, 0)
    """
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    """Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding

    Algorithm).
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        # Save historical weights to check convergence criteria
        old_weights = weights.copy()

        # 1. Compute current predictions
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y

        # 2. Calculate MSE gradients
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        # 3. Apply Gradient Step on smooth MSE component
        w_temp = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # 4. Apply Proximal Step (Soft-Thresholding) strictly to weights
        weights = soft_threshold(w_temp, learning_rate * alpha)

        # 5. Check convergence tolerance
        if np.linalg.norm(weights - old_weights) < tol:
            break

    return weights, bias