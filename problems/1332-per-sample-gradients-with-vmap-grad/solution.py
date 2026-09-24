import jax
import jax.numpy as jnp

def per_sample_grads(w, b, X, y):
    """Per-example gradients of (w·x + b - y)^2.
    Returns (dW, db) with shapes (N, D) and (N,)."""
    def single_loss(w, b, x, yi):
        return (jnp.dot(w, x) + b - yi) ** 2

    grad_fn = jax.grad(single_loss, argnums=(0, 1))
    dW, db = jax.vmap(grad_fn, in_axes=(None, None, 0, 0))(w, b, X, y)
    return dW, db