import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for a Transformer layer using sine and cosine functions.
    
    Args:
        position: int, the total sequence length
        d_model: int, the dimensionality of the model embedding space
        
    Returns:
        numpy.ndarray of shape (position, d_model) and type float16, 
        or -1 for invalid dimensions.
    """
    if position <= 0 or d_model <= 0:
        return -1

    # Create an empty 2D placeholder matrix matching the exact expected test shape
    pe = np.zeros((position, d_model))
    
    for pos in range(position):
        for i in range(d_model):
            # Calculate the explicit scaling frequency denominator for the current column channel
            # We use (i // 2) to ensure that the sine and adjacent cosine share identical frequencies
            denom = 10000.0 ** ((2 * (i // 2)) / d_model)
            angle = pos / denom
            
            if i % 2 == 0:
                pe[pos, i] = np.sin(angle)
            else:
                pe[pos, i] = np.cos(angle)
                
    pos_encoding = np.float16(pe)
    return pos_encoding

