import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    N = tensor.shape[0]
    num_blocks = N // block_size
    
    # FP8-E4M3 maximum representable value
    FP8_MAX = 448.0
    
    # Reshape tensor into blocks
    tensor_reshaped = tensor.reshape(num_blocks, block_size)
    
    # Compute per-block max absolute value
    block_max_abs = np.max(np.abs(tensor_reshaped), axis=1, keepdims=True)
    
    # Compute per-block scale: max_abs / FP8_MAX
    # Add epsilon to avoid division by zero for all-zero blocks
    scales = block_max_abs / FP8_MAX
    scales = np.maximum(scales, 1e-12)  # Avoid zero scales
    
    # Quantize: divide by scale and round to nearest integer
    quantized_reshaped = tensor_reshaped / scales
    
    # Clip to valid FP8-E4M3 range [-448, 448]
    quantized_reshaped = np.clip(quantized_reshaped, -FP8_MAX, FP8_MAX)
    
    # Round to nearest integer and convert to float (not int)
    quantized_reshaped = np.round(quantized_reshaped).astype(np.float64)
    
    # Flatten back to original shape
    quantized = quantized_reshaped.flatten()
    scales = scales.flatten()
    
    return quantized, scales


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    N = quantized.shape[0]
    num_blocks = N // block_size
    
    # Reshape quantized values into blocks
    quantized_reshaped = quantized.reshape(num_blocks, block_size)
    
    # Reshape scales to (num_blocks, 1) for broadcasting
    scales_reshaped = scales.reshape(num_blocks, 1)
    
    # Dequantize: multiply by scale
    dequantized_reshaped = quantized_reshaped.astype(np.float64) * scales_reshaped
    
    # Flatten back to original shape
    dequantized = dequantized_reshaped.flatten()
    
    return dequantized