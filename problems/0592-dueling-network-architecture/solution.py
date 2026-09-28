import numpy as np

def dueling_network_forward(x, shared_weights, shared_bias,
                           value_weights, value_bias,
                           advantage_weights, advantage_bias,
                           aggregation='mean'):
    """
    Forward pass of a Dueling Network Architecture.
    
    Args:
        x: Input features, shape (batch_size, input_dim)
        shared_weights: Shared layer weights, shape (input_dim, hidden_dim)
        shared_bias: Shared layer bias, shape (hidden_dim,)
        value_weights: Value stream weights, shape (hidden_dim, 1)
        value_bias: Value stream bias, shape (1,)
        advantage_weights: Advantage stream weights, shape (hidden_dim, num_actions)
        advantage_bias: Advantage stream bias, shape (num_actions,)
        aggregation: 'mean' or 'max' for advantage centering
    
    Returns:
        Q-values as numpy array, shape (batch_size, num_actions)
    """
    # 1. Shared Feature Layer with ReLU activation
    hidden = np.maximum(0.0, x @ shared_weights + shared_bias)
    
    # 2. Value Stream (shape: batch_size, 1)
    V = hidden @ value_weights + value_bias
    
    # 3. Advantage Stream (shape: batch_size, num_actions)
    A = hidden @ advantage_weights + advantage_bias
    
    # 4. Advantage Centering (identifiability constraint)
    if aggregation == 'mean':
        A_centered = A - np.mean(A, axis=1, keepdims=True)
    elif aggregation == 'max':
        A_centered = A - np.max(A, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported aggregation type: '{aggregation}'. Use 'mean' or 'max'.")
        
    # 5. Recombine to form final Q-values via broadcasting
    Q = V + A_centered
    
    return Q