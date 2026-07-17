import numpy as np

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected (list or np.ndarray)
    :param L: The line vector defining the direction of projection (list or np.ndarray)
    :return: List representing the projection of v onto L rounded to 3 decimal places
    """
    vec_v = np.array(v, dtype=np.float64)
    vec_L = np.array(L, dtype=np.float64)
    
    # Calculate the dot product components
    dot_v_L = np.dot(vec_v, vec_L)
    dot_L_L = np.dot(vec_L, vec_L)
    
    # Handle the edge case of projection onto a zero vector
    if dot_L_L == 0:
        return [round(0.0, 3) for _ in L]
        
    # Scale vector L by the projection scalar factor
    projection = (dot_v_L / dot_L_L) * vec_L
    
    # Convert back to a python list rounded to three decimal places
    return [round(float(x), 3) for x in projection]