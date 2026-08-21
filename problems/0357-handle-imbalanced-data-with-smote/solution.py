import numpy as np

def smote(X_minority, n_synthetic, k=5, random_seed=42):
    """
    Generate synthetic samples using SMOTE algorithm.
    
    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider
        random_seed: Random seed for reproducibility (used for torch compatibility)
        
    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    # Convert to numpy array if it's a tensor
    if hasattr(X_minority, 'numpy'):
        X_array = X_minority.numpy()
    else:
        X_array = np.array(X_minority, dtype=np.float64)
    
    n_samples, n_features = X_array.shape
    
    # If no samples or no synthetic samples needed, return empty array
    if n_samples == 0 or n_synthetic == 0:
        return np.array([], dtype=np.float64).reshape(0, n_features)
    
    # k_actual = min(k, n_samples - 1)
    k_actual = min(k, n_samples - 1)
    
    # If k_actual is 0, return empty array
    if k_actual == 0:
        return np.array([], dtype=np.float64).reshape(0, n_features)
    
    # Initialize synthetic samples
    synthetic_samples = np.zeros((n_synthetic, n_features), dtype=np.float64)
    
    # Generate synthetic samples one at a time
    for idx in range(n_synthetic):
        # 1. Select a base sample randomly
        base_idx = np.random.randint(0, n_samples)
        x_i = X_array[base_idx]
        
        # 2. Find k_actual nearest neighbors (excluding x_i itself)
        # Compute Euclidean distances to all samples
        distances = np.sqrt(np.sum((X_array - x_i) ** 2, axis=1))
        
        # Set distance to itself to infinity to exclude it
        distances[base_idx] = np.inf
        
        # Get indices of k_actual nearest neighbors
        neighbor_indices = np.argsort(distances)[:k_actual]
        
        # 3. Select a random neighbor from the k_actual nearest
        neighbor_idx_in_k = np.random.randint(0, k_actual)
        neighbor_idx = neighbor_indices[neighbor_idx_in_k]
        x_nn = X_array[neighbor_idx]
        
        # 4. Generate random gap in [0, 1)
        gap = np.random.random()
        
        # Create synthetic sample through interpolation
        synthetic_samples[idx] = x_i + gap * (x_nn - x_i)
    
    return synthetic_samples