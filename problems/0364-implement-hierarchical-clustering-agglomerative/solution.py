import torch
import numpy as np

def agglomerative_clustering(X: list, n_clusters: int, linkage: str = 'single') -> torch.Tensor:
    """
    Perform agglomerative hierarchical clustering.
    
    Args:
        X: Data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: 'single', 'complete', or 'average'
    
    Returns:
        torch.Tensor of cluster labels for each sample
    """
    X = np.array(X, dtype=float)
    n_samples = X.shape[0]
    
    # Initialize each sample as its own cluster, tracking the points contained in each cluster
    clusters = {i: [i] for i in range(n_samples)}
    
    # Pre-compute pairwise Euclidean distances between individual data points
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    def compute_cluster_distance(c1_indices, c2_indices):
        sub_dists = dist_matrix[np.ix_(c1_indices, c2_indices)]
        if linkage == 'single':
            return np.min(sub_dists)
        elif linkage == 'complete':
            return np.max(sub_dists)
        elif linkage == 'average':
            return np.mean(sub_dists)
        else:
            raise ValueError(f"Unknown linkage: {linkage}")

    while len(clusters) > n_clusters:
        cluster_ids = sorted(list(clusters.keys()))
        n_curr = len(cluster_ids)
        
        best_dist = float('inf')
        best_pair = None
        
        # Find the pair of clusters to merge according to linkage and tie-breaking rules
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]
                
                d = compute_cluster_distance(clusters[c1], clusters[c2])
                
                # Check for minimum distance with strict tie-breaking:
                # 1. Smaller distance
                # 2. First cluster has smallest index
                # 3. Second cluster has smallest index
                if d < best_dist - 1e-9:
                    best_dist = d
                    best_pair = (c1, c2)
                elif abs(d - best_dist) <= 1e-9:
                    if best_pair is None or c1 < best_pair[0] or (c1 == best_pair[0] and c2 < best_pair[1]):
                        best_pair = (c1, c2)
                        
        c1, c2 = best_pair
        # Always merge into the cluster with the smaller index
        target_c = min(c1, c2)
        source_c = max(c1, c2)
        
        clusters[target_c].extend(clusters[source_c])
        del clusters[source_c]
        
    # Assign final cluster labels based on the sorted order of remaining cluster indices
    remaining_clusters = sorted(list(clusters.keys()))
    labels = np.zeros(n_samples, dtype=int)
    
    for new_label, c_id in enumerate(remaining_clusters):
        for sample_idx in clusters[c_id]:
            labels[sample_idx] = new_label
            
    return torch.tensor(labels, dtype=torch.long)