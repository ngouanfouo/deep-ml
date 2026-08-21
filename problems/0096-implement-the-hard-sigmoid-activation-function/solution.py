def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    # Keras implementation: hard_sigmoid(x) = max(0, min(1, 0.2*x + 0.5))
    output = 0.2 * x + 0.5
    
    # Clip values to [0, 1]
    if output < 0:
        return 0.0
    elif output > 1:
        return 1.0
    else:
        return output