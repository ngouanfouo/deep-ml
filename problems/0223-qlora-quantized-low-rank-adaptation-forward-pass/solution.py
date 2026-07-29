import numpy as np

def qlora_forward(
    x: list[list[float]],
    quantized_W: list[list[int]],
    scale: float,
    zero_point: float,
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    
    Args:
        x: Input matrix (batch_size x in_features)
        quantized_W: 4-bit quantized weights (in_features x out_features)
                     Values are integers that need to be dequantized
        scale: Quantization scale factor
        zero_point: Quantization zero point for dequantization
        A: LoRA matrix A (rank x out_features) - full precision
        B: LoRA matrix B (in_features x rank) - full precision
        alpha: LoRA scaling factor
        
    Returns:
        Output matrix (batch_size x out_features)
    """
    # Convert to numpy arrays
    x = np.array(x, dtype=np.float64)
    quantized_W = np.array(quantized_W, dtype=np.float64)
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    
    # Dequantize W: W = quantized_W * scale + zero_point
    W = quantized_W * scale + zero_point
    
    # Determine rank from A and B dimensions
    # B: in_features x rank, A: rank x out_features
    rank = B.shape[1]  # or A.shape[0]
    
    # Compute scaling factor: alpha / rank
    scaling = alpha / rank
    
    # Pretrained path with dequantized weights
    pretrained = x @ W
    
    # LoRA adaptation path: x @ B @ A
    lora_output = x @ B @ A
    
    # Combine with scaling factor
    output = pretrained + scaling * lora_output
    
    return output.tolist()