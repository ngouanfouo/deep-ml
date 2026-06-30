import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling.
    
    Args:
        x: list of float values to quantize
        block_size: number of elements per scaling block
        
    Returns:
        dict with keys:
            'quantized': list of dequantized values (rounded to 4 decimals)
            'scales': list of per-block scale factors (rounded to 4 decimals)
    """
    # FP4 E2M1 representable positive values (including zero)
    fp4_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # Include negative counterparts for quantization
    fp4_all = np.concatenate([-fp4_values[1:][::-1], fp4_values])
    
    # Convert input to numpy array for easier processing
    x_arr = np.array(x)
    n = len(x_arr)
    
    # Pad the array if needed
    if n % block_size != 0:
        pad_size = block_size - (n % block_size)
        x_padded = np.concatenate([x_arr, np.zeros(pad_size)])
    else:
        x_padded = x_arr
        pad_size = 0
    
    num_blocks = len(x_padded) // block_size
    quantized_padded = np.zeros_like(x_padded)
    scales = []
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        block = x_padded[start_idx:end_idx]
        
        # Find maximum absolute value in the block
        amax = np.max(np.abs(block))
        
        # Compute scale factor (power of 2)
        if amax == 0:
            scale = 1.0
        else:
            # raw_scale = amax / 6.0 (since 6.0 is max FP4 representable magnitude)
            raw_scale = amax / 6.0
            # Round up to nearest power of 2
            # We want: 2^k >= raw_scale, where k is integer
            # k = ceil(log2(raw_scale))
            if raw_scale <= 0:
                scale = 1.0
            else:
                k = int(np.ceil(np.log2(raw_scale)))
                scale = 2.0 ** k
        
        scales.append(scale)
        
        # Quantize each element in the block
        for i, val in enumerate(block):
            if val == 0:
                quantized_padded[start_idx + i] = 0.0
            else:
                # Scale the value
                scaled_val = val / scale
                
                # Find nearest FP4 value
                # Handle sign
                sign = np.sign(scaled_val)
                abs_scaled = np.abs(scaled_val)
                
                # Find nearest representable value
                # For tie-breaking, choose the one with smaller magnitude
                distances = np.abs(fp4_values - abs_scaled)
                min_dist_idx = np.argmin(distances)
                
                # Check for ties (values equally close)
                min_dist = distances[min_dist_idx]
                # Find all indices with the same minimum distance
                tie_indices = np.where(np.abs(distances - min_dist) < 1e-10)[0]
                
                if len(tie_indices) > 1:
                    # Break ties by choosing value with smaller magnitude
                    # Among tied values, pick the smallest magnitude
                    tied_values = fp4_values[tie_indices]
                    nearest_abs = np.min(tied_values)
                else:
                    nearest_abs = fp4_values[min_dist_idx]
                
                # Apply sign
                nearest = sign * nearest_abs
                
                # Dequantize by multiplying back by scale
                quantized_padded[start_idx + i] = nearest * scale
    
    # Remove padding
    if pad_size > 0:
        quantized = quantized_padded[:-pad_size]
    else:
        quantized = quantized_padded
    
    # Round to 4 decimal places
    quantized = np.round(quantized, 4)
    scales = np.round(scales, 4)
    
    return {
        'quantized': quantized.tolist(),
        'scales': scales.tolist()
    }