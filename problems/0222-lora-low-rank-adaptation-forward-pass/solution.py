import numpy as np

def lora_forward(
    x: list[list[float]],
    W: list[list[float]],
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    Compute the LoRA forward pass.
    
    Args:
        x: Input matrix (batch_size x in_features)
        W: Frozen pretrained weights (in_features x out_features)
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output matrix (batch_size x out_features)
    """
    # Convert to numpy arrays
    x = np.array(x, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    
    # Determine rank from A and B dimensions
    # B: in_features x rank, A: rank x out_features
    rank = B.shape[1]  # or A.shape[0]
    
    # Compute scaling factor: alpha / rank
    scaling = alpha / rank
    
    # Forward pass: y = x @ W + scaling * (x @ B @ A)
    # Note: In original LoRA implementation, the order is typically x @ A @ B
    # But here we have B: in_features x rank, A: rank x out_features
    # So x @ B gives (batch_size x rank), then @ A gives (batch_size x out_features)
    
    # Pretrained path
    pretrained = x @ W
    
    # LoRA adaptation path: x @ B @ A
    lora_output = x @ B @ A
    
    # Combine with scaling factor
    output = pretrained + scaling * lora_output
    
    return output.tolist()