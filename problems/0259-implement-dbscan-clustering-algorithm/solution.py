import torch

def dbscan(X: torch.Tensor, eps: float, min_samples: int) -> torch.Tensor:
    """
    Implement DBSCAN clustering algorithm.
    
    Parameters:
    - X: 2D torch tensor of shape (n_samples, n_features)
    - eps: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood for a core point
    
    Returns:
    - labels: 1D torch tensor of cluster labels (-1 for noise points)
    """
    n_samples = X.shape[0]
    labels = torch.full((n_samples,), -1, dtype=torch.long)
    visited = torch.zeros(n_samples, dtype=torch.bool)
    cluster_id = 0
    
    def get_neighbors(idx):
        """Get indices of all points within eps distance of point idx."""
        # Compute distances from point idx to all other points
        diff = X - X[idx]  # (n_samples, n_features)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
        return torch.where(dist <= eps)[0]
    
    def expand_cluster(point_idx, neighbors):
        """Expand cluster from a core point."""
        nonlocal cluster_id
        
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            current_point = neighbors[i].item()
            
            if visited[current_point]:
                i += 1
                continue
            
            visited[current_point] = True
            current_neighbors = get_neighbors(current_point)
            
            if len(current_neighbors) >= min_samples:
                for neighbor_idx in current_neighbors:
                    neighbor_idx = neighbor_idx.item()
                    if labels[neighbor_idx] == -1 and neighbor_idx not in neighbors:
                        neighbors = torch.cat([neighbors, torch.tensor([neighbor_idx], dtype=torch.long)])
            
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            
            i += 1
    
    # Process each point
    for point_idx in range(n_samples):
        if visited[point_idx]:
            continue
        
        visited[point_idx] = True
        neighbors = get_neighbors(point_idx)
        
        if len(neighbors) >= min_samples:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1
    
    return labels