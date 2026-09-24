import jax
import jax.numpy as jnp

def fit_line(x, y, lr, steps):
    """Gradient-descent fit of y ≈ w*x + b from w=b=0.
    Returns (w, b, final_loss) as Python floats."""
    def loss(w, b):
        preds = w * x + b
        return jnp.mean((preds - y) ** 2)

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1))

    w = 0.0
    b = 0.0

    for _ in range(steps):
        loss_val, (dw, db) = value_and_grad(w, b)
        w = w - lr * dw
        b = b - lr * db

    final_loss = loss(w, b)
    return float(w), float(b), float(final_loss)