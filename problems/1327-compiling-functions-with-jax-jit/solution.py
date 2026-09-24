import jax
import jax.numpy as jnp

@jax.jit
def standardize(x):
    """Return (x - mean(x)) / std(x) for a 1-D array x."""
    return (x - jnp.mean(x)) / jnp.std(x)