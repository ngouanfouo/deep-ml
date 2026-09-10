import numpy as np

def concat_layer_features(features):
    """
    Concatenate per-layer ViT feature maps along the feature dimension.

    Args:
        features: list of np.ndarray, each of shape (num_patches, hidden_dim_i)

    Returns:
        list: nested list representing array of shape (num_patches, sum hidden_dim_i)
    """
    if len(features) == 0:
        return []
    
    concatenated = np.concatenate(features, axis=1)
    return concatenated.tolist()