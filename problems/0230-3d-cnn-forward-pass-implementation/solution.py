import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Perform 3D convolution forward pass.
    
    Slide a 3D kernel over input volume, computing dot products.
    
    Args:
        input_volume: Shape (C, D, H, W)
          C = channels, D = depth/time, H = height, W = width
        kernel: Shape (C, kD, kH, kW)
          Must match input channels
        stride: (stride_d, stride_h, stride_w)
          Step size in each dimension
        padding: (pad_d, pad_h, pad_w)
          Zero-padding in each dimension
    
    Returns:
        Output volume: Shape (1, D_out, H_out, W_out)
          Single output channel
    """
    C, D, H, W = input_volume.shape
    C_k, kD, kH, kW = kernel.shape
    
    if C != C_k:
        raise ValueError("Input channels and kernel channels must match.")
        
    pad_d, pad_h, pad_w = padding
    stride_d, stride_h, stride_w = stride
    
    # 1. Apply zero-padding to the spatial dimensions (D, H, W)
    # No padding is applied to the Channel dimension (index 0)
    padded_input = np.pad(
        input_volume,
        ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )
    
    # Get padded dimensions
    _, D_pad, H_pad, W_pad = padded_input.shape
    
    # 2. Calculate output dimensions
    D_out = (D_pad - kD) // stride_d + 1
    H_out = (H_pad - kH) // stride_h + 1
    W_out = (W_pad - kW) // stride_w + 1
    
    # Initialize the output volume with a single channel
    output_volume = np.zeros((1, D_out, H_out, W_out), dtype=float)
    
    # 3. Perform the 3D Convolution
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                # Calculate the slice boundaries for the current window patch
                d_start = d * stride_d
                d_end = d_start + kD
                
                h_start = h * stride_h
                h_end = h_start + kH
                
                w_start = w * stride_w
                w_end = w_start + kW
                
                # Extract the 3D patch across all channels
                patch = padded_input[:, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Compute element-wise product and sum across all dimensions
                output_volume[0, d, h, w] = np.sum(patch * kernel)
                
    return output_volume

