import torch

def silhouette_score(X: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the Silhouette Score for clustering evaluation.
    
    X: shape (n_samples, n_features) - data points
    labels: shape (n_samples,) - cluster label for each point
    
    Returns: silhouette score (float between -1 and 1)
    """
    n_samples = X.shape[0]
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    
    # Edge cases
    if n_clusters <= 1 or n_samples == n_clusters:
        return 0.0
    
    # Compute pairwise distances
    # Using broadcasting to compute all pairwise distances efficiently
    # X: (n_samples, n_features)
    # X.unsqueeze(0): (1, n_samples, n_features)
    # X.unsqueeze(1): (n_samples, 1, n_features)
    # diff: (n_samples, n_samples, n_features)
    diff = X.unsqueeze(0) - X.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(dim=2))  # (n_samples, n_samples)
    
    # For each sample, compute a(i) and b(i)
    silhouette_scores = torch.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i]
        
        # Same cluster distances (excluding self)
        same_cluster_mask = (labels == label_i)
        same_cluster_dist = distances[i, same_cluster_mask]
        same_cluster_dist = same_cluster_dist[same_cluster_dist > 0]  # exclude self
        
        if len(same_cluster_dist) > 0:
            a_i = same_cluster_dist.mean()
        else:
            # If this is the only point in the cluster, set a_i = 0
            a_i = 0.0
        
        # Different clusters distances
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster_mask = (labels == other_label)
            other_cluster_dist = distances[i, other_cluster_mask]
            if len(other_cluster_dist) > 0:
                mean_dist = other_cluster_dist.mean()
                b_i = min(b_i, mean_dist)
        
        # If b_i is still inf, this means there are no other clusters
        if b_i == float('inf'):
            silhouette_scores[i] = 0.0
        elif a_i == 0 and b_i == 0:
            silhouette_scores[i] = 0.0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    # Return mean silhouette score rounded to 4 decimal places
    return round(silhouette_scores.mean().item(), 4)