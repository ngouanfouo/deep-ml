import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    """
    Perform forward pass of a DenseNet dense block.
    
    Args:
        input_data: Input tensor of shape (N, H, W, C0)
        num_layers: Number of dense layers to apply
        growth_rate: Number of channels added by each layer
        kernels: List of convolution kernels, one per layer
        kernel_size: Tuple (kh, kw) for kernel size (default (3, 3))
    
    Returns:
        Tensor of shape (N, H, W, C0 + num_layers * growth_rate)
    
    Raises:
        ValueError: If kernel input channel dimension doesn't match current feature maps
    """
    # Validate input
    if not isinstance(kernels, list) or len(kernels) != num_layers:
        raise ValueError(f"Expected {num_layers} kernels, got {len(kernels)}")
    
    # Get input dimensions
    N, H, W, C0 = input_data.shape
    kh, kw = kernel_size
    
    # Initialize running tensor with input data
    running = input_data.copy()
    current_channels = C0
    
    # Apply each dense layer
    for layer_idx in range(num_layers):
        kernel = kernels[layer_idx]
        
        # Validate kernel shape
        expected_in_channels = current_channels
        expected_kernel_shape = (kh, kw, expected_in_channels, growth_rate)
        if kernel.shape != expected_kernel_shape:
            raise ValueError(
                f"Kernel {layer_idx} has shape {kernel.shape}, "
                f"expected {expected_kernel_shape} (kh={kh}, kw={kw}, "
                f"in_channels={expected_in_channels}, out_channels={growth_rate})"
            )
        
        # Apply ReLU activation to running tensor
        activated = np.maximum(0, running)
        
        # Perform convolution with padding to preserve H and W
        # Calculate padding for 'same' convolution
        pad_h = kh // 2
        pad_w = kw // 2
        
        # Pad the input
        padded = np.pad(activated, 
                        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                        mode='constant')
        
        # Perform convolution
        # Initialize output tensor
        conv_output = np.zeros((N, H, W, growth_rate))
        
        # For each batch sample
        for n in range(N):
            # For each output channel
            for out_c in range(growth_rate):
                # For each spatial position
                for i in range(H):
                    for j in range(W):
                        # Extract the patch
                        patch = padded[n, i:i+kh, j:j+kw, :]
                        # Apply kernel (element-wise multiply and sum)
                        conv_output[n, i, j, out_c] = np.sum(patch * kernel[:, :, :, out_c])
        
        # Concatenate convolution output along channel axis
        running = np.concatenate([running, conv_output], axis=-1)
        current_channels += growth_rate
    
    return running