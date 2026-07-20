import numpy as np

def rectified_flow_step(
    X_1: np.ndarray,
    X_0: np.ndarray,
    t: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    lr: float
) -> dict:
    """
    Perform one training step for a rectified flow model.

    Args:
        X_1: Data samples, shape (N, D)
        X_0: Noise samples, shape (N, D)
        t: Timesteps in [0, 1], shape (N, 1)
        W: Velocity network weights, shape (D+1, D)
        b: Velocity network bias, shape (1, D)
        lr: Learning rate for gradient descent

    Returns:
        Dictionary containing x_t, v_target, v_pred, loss, W_new, and b_new.
    """
    N, D = X_1.shape

    # 1. Interpolation: Straight-line path from noise (t=0) to data (t=1)
    x_t = t * X_1 + (1.0 - t) * X_0

    # 2. Target Computation: Constant velocity along the path
    v_target = X_1 - X_0

    # 3. Forward Pass
    # Concatenate the interpolated sample and timestep along the feature dimension
    v_input = np.hstack([x_t, t])  # Shape: (N, D+1)
    v_pred = np.matmul(v_input, W) + b  # Shape: (N, D)

    # 4. Loss Computation: Mean Squared Error (MSE) across all dimensions and samples
    error = v_pred - v_target
    loss = float(np.mean(error ** 2))

    # 5. Backward Pass (Analytical Gradients)
    # The derivative of the loss with respect to v_pred is (2 / (N * D)) * error
    d_vpred = (2.0 / (N * D)) * error
    
    # Gradients for weights and bias
    dW = np.matmul(v_input.T, d_vpred)   # Shape: (D+1, D)
    db = np.sum(d_vpred, axis=0, keepdims=True)  # Shape: (1, D)

    # 6. Gradient Descent Parameter Update
    W_new = W - lr * dW
    b_new = b - lr * db

    return {
        'x_t': x_t,
        'v_target': v_target,
        'v_pred': v_pred,
        'loss': loss,
        'W_new': W_new,
        'b_new': b_new
    }