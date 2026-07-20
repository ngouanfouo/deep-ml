import numpy as np

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient arrays (can be structured as lists of lists/iterables)
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient arrays maintaining the original structure
    """
    # 1. Compute the squared sum across all parameters globally
    total_norm_sq = 0.0
    for grad in gradients:
        total_norm_sq += np.sum(np.square(grad))
        
    # 2. Calculate the global L2 norm
    global_norm = np.sqrt(total_norm_sq)
    
    # 3. If global norm is within bounds, return original values as float lists
    if global_norm <= max_norm:
        return [[float(g) for g in grad] for grad in gradients]
        
    # 4. Otherwise, compute the down-scaling multiplier factor safely
    # Adding a small epsilon protection prevents division by zero if global_norm is 0
    scale_factor = max_norm / (global_norm + 1e-6)
    
    # 5. Rescale all parameters proportionally and restore the nested structure
    clipped_gradients = []
    for grad in gradients:
        clipped_grad = [float(g * scale_factor) for g in grad]
        clipped_gradients.append(clipped_grad)
        
    return clipped_gradients