import torch

def int8_quantize(x: list[float]) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array.
    
    Symmetric quantization:
    - Scale factor: scale = max(|x|) / 127
    - Quantize: q = round(x / scale)
    - Dequantize: x_reconstructed = q * scale
    
    The quantization is symmetric around zero, meaning zero in floating-point
    maps to zero in integer representation.
    
    Args:
        x: Input list of floating-point values
        
    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
        
    Notes:
        - The INT8 range is [-127, 127] (not using -128 to maintain symmetry)
        - Rounding uses half-to-even (PyTorch's default)
        - Dequantized values may have rounding errors
        - If all values are zero, scale is set to 1.0
    """
    # Convert to tensor if needed
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.tensor(x, dtype=torch.float32)
    else:
        x_tensor = x.float()
    
    # Get device and dtype
    device = x_tensor.device
    
    # Check if tensor is empty
    if x_tensor.numel() == 0:
        return {
            'quantized': [],
            'scale': 1.0,
            'dequantized': []
        }
    
    # Handle edge case: all zeros
    if torch.all(x_tensor == 0):
        quantized = torch.zeros_like(x_tensor, dtype=torch.int8)
        scale = torch.tensor(1.0, device=device)
        dequantized = torch.zeros_like(x_tensor, dtype=torch.float32)
    else:
        # Calculate max absolute value
        abs_max = torch.max(torch.abs(x_tensor))
        
        # Calculate scale factor
        scale = abs_max / 127.0
        
        # Quantize with proper rounding
        # Note: torch.round uses half-to-even rounding
        quantized_raw = x_tensor / scale
        quantized = torch.round(quantized_raw).clamp(-127, 127).to(torch.int8)
        
        # Dequantize
        dequantized = quantized.float() * scale
    
    # Convert to lists with proper rounding
    quantized_list = quantized.tolist()
    scale_value = float(scale.cpu().item())
    scale_rounded = round(scale_value, 6)
    dequantized_list = [round(float(v), 4) for v in dequantized.cpu()]
    
    return {
        'quantized': quantized_list,
        'scale': scale_rounded,
        'dequantized': dequantized_list
    }