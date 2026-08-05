import torch

def calinski_harabasz_score(X: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering evaluation.
    
    Args:
        X: torch tensor of shape (n_samples, n_features) containing data points
        labels: torch tensor of shape (n_samples,) containing cluster assignments
    
    Returns:
        float: Calinski-Harabasz score (higher is better)
    """
    n_samples = X.shape[0]
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    
    # Edge cases
    if n_clusters <= 1 or n_samples == n_clusters:
        return 0.0
    
    # Global centroid
    global_centroid = X.mean(dim=0)  # (n_features,)
    
    # Compute within-cluster dispersion (sum of squared distances to cluster centroids)
    within_dispersion = 0.0
    # Compute between-cluster dispersion (weighted sum of squared distances between centroids and global centroid)
    between_dispersion = 0.0
    
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        cluster_size = cluster_points.shape[0]
        
        # Cluster centroid
        cluster_centroid = cluster_points.mean(dim=0)
        
        # Within-cluster dispersion: sum of squared distances from points to cluster centroid
        diff_within = cluster_points - cluster_centroid
        within_dispersion += (diff_within ** 2).sum()
        
        # Between-cluster dispersion: cluster_size * squared distance from cluster centroid to global centroid
        diff_between = cluster_centroid - global_centroid
        between_dispersion += cluster_size * (diff_between ** 2).sum()
    
    # Calinski-Harabasz Index formula
    # CH = (B / (k-1)) / (W / (n-k))
    # where B = between_cluster_dispersion, W = within_cluster_dispersion
    # k = n_clusters, n = n_samples
    
    # Avoid division by zero
    if within_dispersion == 0:
        return 0.0
    
    numerator = between_dispersion / (n_clusters - 1)
    denominator = within_dispersion / (n_samples - n_clusters)
    score = numerator / denominator
    
    return float(score.item())