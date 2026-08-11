import numpy as np

def tdnn_forward(sequences, layer_configs):
    """
    Forward pass through a multi-layer Time-Delay Neural Network.

    Args:
        sequences: list of numpy arrays, each shape (T_i, D_input)
        layer_configs: list of dicts with keys:
            'weights': np.ndarray of shape (len(offsets)*D_in, D_out)
            'bias': np.ndarray of shape (D_out,)
            'offsets': list of int time offsets
            'activation': 'relu' or 'none'

    Returns:
        List of nested Python lists, one per input sequence,
        with values rounded to 4 decimal places.
    """
    # Process each sequence independently
    results = []
    
    for seq in sequences:
        # Start with the input sequence
        current_seq = seq.copy()  # shape: (T, D)
        
        # Track if sequence became invalid (too short)
        valid = True
        
        # Apply each layer
        for layer_config in layer_configs:
            if not valid:
                break
                
            weights = layer_config['weights']
            bias = layer_config['bias']
            offsets = layer_config['offsets']
            activation = layer_config['activation']
            
            T, D_in = current_seq.shape
            D_out = weights.shape[1]
            
            # Calculate valid time steps where all offsets are within bounds
            min_offset = min(offsets)
            max_offset = max(offsets)
            
            # Valid time steps: from -min_offset to T-1-max_offset (inclusive)
            start_idx = -min_offset  # First valid index
            end_idx = T - max_offset - 1  # Last valid index
            
            if start_idx > end_idx:
                # Sequence is too short for this layer
                valid = False
                break
            
            # Number of valid output time steps
            T_out = end_idx - start_idx + 1
            
            # Initialize output array
            output = np.zeros((T_out, D_out))
            
            # Process each valid time step
            for t_out, t_in in enumerate(range(start_idx, end_idx + 1)):
                # Gather context vectors
                context_vectors = []
                for offset in offsets:
                    context_vectors.append(current_seq[t_in + offset])
                
                # Concatenate context vectors
                concat_vector = np.concatenate(context_vectors)
                
                # Apply linear transformation
                linear_out = concat_vector @ weights + bias
                
                # Apply activation function
                if activation == 'relu':
                    output[t_out] = np.maximum(0, linear_out)
                else:  # 'none'
                    output[t_out] = linear_out
            
            # Update current sequence for next layer
            current_seq = output
        
        # After processing all layers or if invalid, prepare result
        if not valid:
            results.append([])
        else:
            # Round to 4 decimal places and convert to list
            rounded_seq = np.round(current_seq, 4)
            results.append(rounded_seq.tolist())
    
    return results