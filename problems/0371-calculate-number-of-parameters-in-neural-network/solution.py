import torch
import torch.nn as nn

def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network
    using PyTorch's built-in nn modules.

    Args:
        layers: List of dictionaries, each describing a layer.
                Supported types: 'dense' (uses nn.Linear), 'conv2d' (uses nn.Conv2d).

    Returns:
        Total number of trainable parameters (int).
    """
    total_params = 0
    
    for layer_cfg in layers:
        layer_type = layer_cfg.get('type')
        has_bias = layer_cfg.get('bias', True)  # Default to True if not specified
        
        if layer_type == 'dense':
            in_features = layer_cfg['input_size']
            out_features = layer_cfg['output_size']
            
            # Instantiate PyTorch Linear module to easily count parameters
            module = nn.Linear(in_features, out_features, bias=has_bias)
            
        elif layer_type == 'conv2d':
            in_channels = layer_cfg['in_channels']
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg['kernel_size']
            
            # Instantiate PyTorch Conv2d module
            module = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=has_bias)
            
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
            
        # Sum up parameters that require gradients (trainable parameters)
        layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += layer_params
        
    return total_params