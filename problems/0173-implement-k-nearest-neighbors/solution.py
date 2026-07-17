import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point.
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples.
        When distances are tied, points appearing earlier in the input list come first.
    """
    # Convert query_point to a numpy array for easy vector operations
    q = np.array(query_point)
    
    def get_distance(p):
        # Calculate the Euclidean distance between point p and query_point q
        return np.linalg.norm(np.array(p) - q)
    
    # Sort the points list using the calculated Euclidean distance as the key.
    # Python's Timsort algorithm is stable, preserving original order on ties.
    sorted_points = sorted(points, key=get_distance)
    
    # Return the first k elements
    return sorted_points[:k]