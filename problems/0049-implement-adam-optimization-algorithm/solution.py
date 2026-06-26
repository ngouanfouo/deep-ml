import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Initialize first and second moment estimates
    m = np.zeros_like(x0)  # First moment (mean of gradients)
    v = np.zeros_like(x0)  # Second moment (uncentered variance of gradients)
    
    # Initialize parameters
    x = x0.copy()
    
    for t in range(1, num_iterations + 1):
        # Compute gradient at current parameters
        g = grad(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g
        
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (g * g)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)
        
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return x