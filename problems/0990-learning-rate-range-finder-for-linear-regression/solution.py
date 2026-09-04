import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    w = w0.astype(np.float64).copy()
    n = X.shape[0]
    losses = []

    for k in range(n_steps):
        # Compute learning rate for this step
        if n_steps == 1:
            lr = 10 ** a
        else:
            lr = 10 ** (a + (b - a) * k / (n_steps - 1))

        # Full-batch MSE gradient
        grad = (2.0 / n) * X.T @ (X @ w - y)

        # Update weights
        w = w - lr * grad

        # Record loss after update
        loss = np.mean((X @ w - y) ** 2)
        losses.append(float(loss))

    return losses