import torch
from typing import Tuple

def train_paris_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train Paris-style decentralized model.

    1. Partition data by cluster_ids
    2. Train each expert on its partition IN ISOLATION (learns mean target)
    3. Train router by computing cluster centroids (learns to route by similarity)

    Args:
        features: torch.Tensor of shape (N, D) - feature vectors
        targets: torch.Tensor of shape (N,) - target values
        cluster_ids: torch.Tensor of shape (N,) with long dtype - cluster assignment per point
        n_clusters: Number of expert clusters

    Returns:
        expert_predictions: torch.Tensor of shape (n_clusters,) - mean target per expert
        router_centroids: torch.Tensor of shape (n_clusters, D) - mean feature per cluster
    """
    features = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long)
    
    n_samples, feature_dim = features.shape
    
    expert_predictions = torch.zeros(n_clusters, dtype=features.dtype, device=features.device)
    router_centroids = torch.zeros((n_clusters, feature_dim), dtype=features.dtype, device=features.device)
    
    for c in range(n_clusters):
        mask = (cluster_ids == c)
        if mask.any():
            expert_predictions[c] = targets[mask].mean()
            router_centroids[c] = features[mask].mean(dim=0)
            
    return expert_predictions, router_centroids


def paris_inference(
    feature: torch.Tensor,
    expert_predictions: torch.Tensor,
    router_centroids: torch.Tensor,
    strategy: str = 'top1'
) -> float:
    """
    Run inference with trained Paris model.

    Router selects expert(s) based on feature similarity to learned centroids.

    Args:
        feature: torch.Tensor of shape (D,) - input feature vector
        expert_predictions: torch.Tensor of shape (n_clusters,) - trained expert outputs
        router_centroids: torch.Tensor of shape (n_clusters, D) - learned cluster centroids
        strategy: 'top1' (select nearest) or 'weighted' (inverse distance weighting)

    Returns:
        Model prediction as a Python float
    """
    feature = torch.as_tensor(feature, dtype=router_centroids.dtype, device=router_centroids.device)
    
    # Compute L2 distance from input feature to all learned cluster centroids
    distances = torch.linalg.norm(router_centroids - feature, dim=-1)
    
    if strategy == 'top1':
        nearest_idx = torch.argmin(distances)
        return float(expert_predictions[nearest_idx].item())
        
    elif strategy == 'weighted':
        # Inverse Distance Weighting (IDW)
        eps = 1e-8
        if torch.any(distances < eps):
            nearest_idx = torch.argmin(distances)
            return float(expert_predictions[nearest_idx].item())
        
        weights = 1.0 / distances
        weights = weights / torch.sum(weights)
        prediction = torch.sum(weights * expert_predictions)
        return float(prediction.item())
        
    else:
        raise ValueError(f"Unknown routing strategy: '{strategy}'. Supported strategies: 'top1', 'weighted'.")