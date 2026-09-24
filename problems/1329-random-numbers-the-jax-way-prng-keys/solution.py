import jax
import jax.numpy as jnp

def sample_pair(seed, shape):
    """Return (a, b): two different standard-normal arrays of `shape`,
    drawn from two subkeys split off PRNGKey(seed)."""
    key = jax.random.PRNGKey(seed)
    key_a, key_b = jax.random.split(key)
    a = jax.random.normal(key_a, shape)
    b = jax.random.normal(key_b, shape)
    return a, b