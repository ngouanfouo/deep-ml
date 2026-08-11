import numpy as np

def local_outlier_factor(X, k):
    """
    Compute the Local Outlier Factor (LOF) score for each point in X.

    Args:
        X: array-like of shape (n, d)
        k: int, neighborhood size

    Returns:
        list of length n with the LOF score of each point
    """
    X = np.array(X)
    n, d = X.shape
    
    # Compute pairwise Euclidean distances
    # Use broadcasting to compute all pairwise distances efficiently
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
    
    # For each point, find the indices of its k nearest neighbors (excluding self)
    # Get sorted indices for each row (excluding the first which is the point itself)
    neighbor_indices = np.zeros((n, k), dtype=int)
    for i in range(n):
        # argsort gives indices sorted by distance
        sorted_idx = np.argsort(distances[i])
        # The first index is the point itself (distance 0)
        # Take the next k indices
        neighbor_indices[i] = sorted_idx[1:k+1]
    
    # Compute k-distance for each point: distance to k-th nearest neighbor
    k_distances = np.zeros(n)
    for i in range(n):
        sorted_dist = np.sort(distances[i])
        # k-th nearest neighbor is at index k (0-indexed, since index 0 is self)
        k_distances[i] = sorted_dist[k]
    
    # Compute reachability distance: reach_dist(p, o) = max(k-distance(o), d(p, o))
    reach_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                reach_dist[i, j] = max(k_distances[j], distances[i, j])
    
    # Compute local reachability density for each point
    # lrd(p) = 1 / mean(reach_dist(p, o) for o in N_k(p))
    lrd = np.zeros(n)
    for i in range(n):
        neighbors = neighbor_indices[i]
        # Sum of reachability distances to neighbors
        sum_reach = 0
        for neighbor in neighbors:
            sum_reach += reach_dist[i, neighbor]
        # lrd = 1 / (sum_reach / k) = k / sum_reach
        if sum_reach > 0:
            lrd[i] = k / sum_reach
        else:
            # If sum_reach is 0, density is infinite (points coincide)
            # But problem states no two points coincide
            lrd[i] = float('inf')
    
    # Compute LOF for each point
    # LOF(p) = mean(lrd(o) / lrd(p) for o in N_k(p))
    lof_scores = np.zeros(n)
    for i in range(n):
        neighbors = neighbor_indices[i]
        if lrd[i] == float('inf'):
            # If lrd(p) is infinite, handle specially
            # If any neighbor also has infinite lrd, LOF should be 1.0
            # Otherwise, LOF should be 0 (or very small)
            all_infinite = True
            for neighbor in neighbors:
                if lrd[neighbor] != float('inf'):
                    all_infinite = False
                    break
            if all_infinite:
                lof_scores[i] = 1.0
            else:
                # If lrd(p) is infinite but neighbors are finite, LOF is 0
                lof_scores[i] = 0.0
        else:
            # Sum of lrd(neighbor) / lrd(i)
            sum_ratio = 0
            for neighbor in neighbors:
                if lrd[neighbor] == float('inf'):
                    # If neighbor has infinite density, the ratio is infinite
                    # This would make LOF infinite
                    sum_ratio = float('inf')
                    break
                else:
                    sum_ratio += lrd[neighbor] / lrd[i]
            
            if sum_ratio == float('inf'):
                lof_scores[i] = float('inf')
            else:
                lof_scores[i] = sum_ratio / k
    
    # Convert to list
    return lof_scores.tolist()