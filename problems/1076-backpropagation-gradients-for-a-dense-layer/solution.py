import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    a_prev = np.asarray(a_prev, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # --- Forward pass ---
    z = W @ a_prev + b                       # (J,)
    a = 1.0 / (1.0 + np.exp(-z))             # (J,)

    # --- Backward pass ---
    # C = sum_j (a_j - y_j)^2  ->  dC/da = 2(a - y)
    dC_da = 2.0 * (a - y)                    # (J,)

    # Sigmoid derivative: a * (1 - a)
    sig_deriv = a * (1.0 - a)                # (J,)

    # dC/dz = dC/da * sigmoid'(z)
    dC_dz = dC_da * sig_deriv                # (J,)

    # Gradients
    dW = np.outer(dC_dz, a_prev)             # (J, K)
    db = dC_dz.copy()                        # (J,)
    da_prev = W.T @ dC_dz                    # (K,)

    return {
        'dW': dW.tolist(),
        'db': db.tolist(),
        'da_prev': da_prev.tolist(),
    }