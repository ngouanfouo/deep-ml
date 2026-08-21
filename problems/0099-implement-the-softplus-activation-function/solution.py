import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x)
    """
    # Handle edge cases to prevent numerical issues
    if x > 50:
        # For large x, log(1 + e^x) ≈ x
        val = x
    elif x < -50:
        # For small x, log(1 + e^x) ≈ e^x
        val = math.exp(x)
    else:
        # For moderate x, compute directly
        val = math.log(1 + math.exp(x))
    
    return round(val, 4)