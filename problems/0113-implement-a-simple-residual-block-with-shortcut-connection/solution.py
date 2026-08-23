import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    # First layer: linear transformation
    h1 = torch.matmul(w1, x)
    # ReLU activation
    h1 = torch.relu(h1)
    
    # Second layer: linear transformation
    h2 = torch.matmul(w2, h1)
    # Add shortcut connection
    h2 = h2 + x
    # Final ReLU activation
    output = torch.relu(h2)
    
    return output