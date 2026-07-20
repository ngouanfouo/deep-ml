import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.

    Args:
        x: input vector
        W1, b1: first linear layer parameters
        W2, b2: second linear layer parameters
        dropout_p: dropout probability
        seed: random seed for reproducibility

    Returns:
        Output vector after FFN block (rounded to 4 decimals)
    """
    # Convert inputs to numpy arrays for easier computation
    x_np = np.array(x)
    W1_np = np.array(W1)
    b1_np = np.array(b1)
    W2_np = np.array(W2)
    b2_np = np.array(b2)
    
    # First linear transformation: h = W1 @ x + b1
    hidden = W1_np @ x_np + b1_np
    
    # Apply ReLU activation: ReLU(h) = max(0, h)
    hidden_relu = np.maximum(0, hidden)
    
    # Second linear transformation: out = W2 @ hidden_relu + b2
    output = W2_np @ hidden_relu + b2_np
    
    # Apply dropout if dropout_p > 0
    if dropout_p > 0:
        # Set random seed for reproducibility
        np.random.seed(seed)
        # Create dropout mask: keep with probability (1 - dropout_p)
        mask = np.random.rand(*output.shape) > dropout_p
        # Scale to maintain expected value during training
        output = output * mask / (1 - dropout_p)
    
    # Add residual connection: output = output + input
    output = output + x_np
    
    # Round to 4 decimal places for reproducibility
    output_rounded = np.round(output, 4)
    
    # Convert back to list
    return output_rounded.tolist()