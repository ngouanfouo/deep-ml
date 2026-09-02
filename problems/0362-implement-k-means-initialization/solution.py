import torch

def kmeans_plus_plus_init(X: torch.Tensor, k: int, seed: int = None) -> torch.Tensor:
    """
    Initialize k centroids using the K-Means++ algorithm.
    
    Args:
        X: Data points of shape (n_samples, n_features) as a torch.Tensor
        k: Number of centroids to initialize
        seed: Random seed for reproducibility
    
    Returns:
        Centroids of shape (k, n_features) as a torch.Tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_samples, n_features = X.shape
    centroids = []
    
    # 1. Choose first centroid uniformly at random
    first_idx = torch.randint(0, n_samples, (1,)).item()
    centroids.append(X[first_idx].unsqueeze(0))  # shape (1, n_features)
    
    # 2. Choose remaining centroids
    for _ in range(1, k):
        # Compute squared distances to the nearest already chosen centroid
        # Stack current centroids: (num_centroids, n_features)
        current_centroids = torch.cat(centroids, dim=0)  # (c, n_features)
        
        # Compute squared distances for all points to all centroids
        # (n_samples, 1, n_features) - (1, c, n_features) -> (n_samples, c, n_features)
        diff = X.unsqueeze(1) - current_centroids.unsqueeze(0)  # (n_samples, c, n_features)
        sq_dists = (diff ** 2).sum(dim=2)  # (n_samples, c)
        
        # Min distance to any centroid per point
        min_dists = sq_dists.min(dim=1)[0]  # (n_samples,)
        
        # Probability proportional to squared distance
        probs = min_dists / min_dists.sum()
        
        # Sample next centroid
        next_idx = torch.multinomial(probs, 1).item()
        centroids.append(X[next_idx].unsqueeze(0))
    
    return torch.cat(centroids, dim=0)