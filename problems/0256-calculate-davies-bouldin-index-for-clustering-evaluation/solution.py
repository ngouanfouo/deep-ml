import numpy as np

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index for clustering evaluation.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features) - data points
    labels: numpy array of shape (n_samples,) - cluster labels
    
    Returns:
    float: Davies-Bouldin Index rounded to 4 decimal places
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Edge case: if there is only one cluster, return 0.0
    if n_clusters <= 1:
        return 0.0
    
    # Compute centroids and scatter for each cluster
    centroids = []
    scatters = []
    
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        
        # Scatter = average Euclidean distance from points to centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        scatter = np.mean(distances)
        scatters.append(scatter)
    
    # Convert to numpy arrays for easier indexing
    centroids = np.array(centroids)
    scatters = np.array(scatters)
    
    # Compute DBI
    # For each cluster, find the maximum similarity ratio with other clusters
    max_ratios = []
    
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i == j:
                continue
            # Distance between centroids
            centroid_dist = np.linalg.norm(centroids[i] - centroids[j])
            # Similarity ratio: (scatter_i + scatter_j) / distance
            if centroid_dist > 0:
                ratio = (scatters[i] + scatters[j]) / centroid_dist
                max_ratio = max(max_ratio, ratio)
        # In case all centroid distances are zero (shouldn't happen with unique clusters)
        if max_ratio == -np.inf:
            max_ratio = 0.0
        max_ratios.append(max_ratio)
    
    # DBI is the average of the maximum ratios
    dbi = np.mean(max_ratios)
    
    return round(dbi, 4)