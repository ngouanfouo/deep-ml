import torch
import torch.nn as nn

def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    total_params = 0
    param_breakdown = {}  # For debugging
    
    for i, layer_dict in enumerate(layers):
        layer_type = layer_dict.get('type')
        layer_name = f"Layer {i+1} ({layer_type})"
        
        if layer_type == 'dense':
            input_size = layer_dict.get('input_size')
            output_size = layer_dict.get('output_size')
            use_bias = layer_dict.get('use_bias', True)
            
            # Calculate parameters
            weight_params = input_size * output_size
            bias_params = output_size if use_bias else 0
            layer_params = weight_params + bias_params
            
            # Store breakdown
            param_breakdown[layer_name] = {
                'weights': weight_params,
                'biases': bias_params,
                'total': layer_params
            }
            
        elif layer_type == 'conv2d':
            in_channels = layer_dict.get('in_channels')
            out_channels = layer_dict.get('out_channels')
            kernel_size = layer_dict.get('kernel_size')
            use_bias = layer_dict.get('use_bias', True)
            
            # Handle kernel_size
            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size
            
            # Calculate parameters
            weight_params = out_channels * in_channels * kernel_h * kernel_w
            bias_params = out_channels if use_bias else 0
            layer_params = weight_params + bias_params
            
            # Store breakdown
            param_breakdown[layer_name] = {
                'weights': weight_params,
                'biases': bias_params,
                'total': layer_params
            }
            
        elif layer_type == 'embedding':
            num_embeddings = layer_dict.get('num_embeddings')
            embedding_dim = layer_dict.get('embedding_dim')
            
            # Calculate parameters
            layer_params = num_embeddings * embedding_dim
            
            # Store breakdown
            param_breakdown[layer_name] = {
                'embedding_table': layer_params,
                'total': layer_params
            }
            
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        
        total_params += layer_params
    
    # Print breakdown (optional)
    # print("\nParameter breakdown:")
    # for layer_name, params in param_breakdown.items():
    #     print(f"  {layer_name}: {params}")
    # print(f"\nTotal parameters: {total_params}")
    
    return total_params