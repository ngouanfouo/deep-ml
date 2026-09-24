import jax
import jax.numpy as jnp

def replace_at(x, indices, values):
    """Return a copy of 1-D array x with x[indices] replaced by values.
    Must not modify x."""
    indices = jnp.asarray(indices)
    values = jnp.asarray(values)
    return x.at[indices].set(values)