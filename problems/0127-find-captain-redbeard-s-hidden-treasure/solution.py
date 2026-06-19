def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.
    
    Returns:
        float: The x-coordinate of the minimum point.
    """
    x = start_x
    velocity = 0
    learning_rate = 0.001
    momentum = 0.9
    max_iterations = 100000
    
    for i in range(max_iterations):
        # f'(x) = 4x^3 - 9x^2
        gradient = 4 * x**3 - 9 * x**2
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * gradient
        x_new = x + velocity
        
        # Check convergence
        if abs(x_new - x) < 1e-8:
            return x_new
            
        x = x_new
    
    return x