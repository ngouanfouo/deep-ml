import numpy as np

def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Perform RMSProp optimization update.
    
    Args:
        params: List of parameter values
        grads: List of gradients for each parameter
        cache: List of cache values (moving average of squared gradients)
        lr: Learning rate
        beta: Decay rate for moving average
        epsilon: Small constant for numerical stability
    
    Returns:
        Tuple of (updated_params, updated_cache)
    """
    # 1. Cast incoming parameter tracking structures to robust NumPy vectors
    p_arr = np.array(params, dtype=np.float64)
    g_arr = np.array(grads, dtype=np.float64)
    c_arr = np.array(cache, dtype=np.float64)
    
    # 2. Update the exponential moving average of squared gradients
    updated_cache = beta * c_arr + (1 - beta) * (g_arr ** 2)
    
    # 3. Apply the learning rate scaling down proportional to the magnitude history
    updated_params = p_arr - (lr / (np.sqrt(updated_cache) + epsilon)) * g_arr
    
    # 4. Coerce data types back to standard float lists for outer environment pipelines
    return list(updated_params.tolist()), list(updated_cache.tolist())