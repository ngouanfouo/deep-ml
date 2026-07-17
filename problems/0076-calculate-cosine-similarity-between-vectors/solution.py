import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity of two vectors.
    Args:
        v1 (numpy.ndarray): 1D array representing the first vector.
        v2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The cosine similarity of the two vectors as a float.
    """
    # 1. Validate that vectors have the same shape
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same shape.")
        
    # 2. Check for empty vectors
    if v1.size == 0:
        raise ValueError("Input vectors cannot be empty.")
        
    # Calculate norms (magnitudes)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 3. Check for zero magnitude vectors to prevent division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Input vectors cannot have zero magnitude.")
        
    # Compute dot product and division
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    
    # Clip the value between -1.0 and 1.0 to handle floating point precision anomalies
    return float(np.clip(similarity, -1.0, 1.0))