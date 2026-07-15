import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    
    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)
    
    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    # Input validation
    if not isinstance(parameter, (int, float, np.ndarray)):
        raise TypeError("parameter must be a number or numpy array")
    if not isinstance(grad, (int, float, np.ndarray)):
        raise TypeError("grad must be a number or numpy array")
    if not isinstance(u, (int, float, np.ndarray)):
        raise TypeError("u must be a number or numpy array")
    if not isinstance(v, (int, float, np.ndarray)):
        raise TypeError("v must be a number or numpy array")
    
    # Ensure consistent types
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Check shapes match
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if parameter.shape != u.shape:
        raise ValueError("parameter and u must have the same shape")
    if parameter.shape != v.shape:
        raise ValueError("parameter and v must have the same shape")
    
    # Update running average of squared gradients: u = rho * u + (1 - rho) * grad^2
    updated_u = rho * u + (1 - rho) * grad * grad
    
    # Compute parameter update: delta_param = -sqrt(v + epsilon) / sqrt(u + epsilon) * grad
    delta_param = -np.sqrt(v + epsilon) / np.sqrt(updated_u + epsilon) * grad
    
    # Update parameter: parameter = parameter + delta_param
    updated_parameter = parameter + delta_param
    
    # Update running average of squared parameter updates: v = rho * v + (1 - rho) * delta_param^2
    updated_v = rho * v + (1 - rho) * delta_param * delta_param
    
    # Return as numpy arrays (or scalar if input was scalar)
    if updated_parameter.size == 1:
        return float(updated_parameter), float(updated_u), float(updated_v)
    else:
        return updated_parameter, updated_u, updated_v