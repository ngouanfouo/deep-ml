import numpy as np

def lle(X: np.ndarray, n_neighbors: int, n_components: int) -> np.ndarray:
    """
    Perform Locally Linear Embedding for dimensionality reduction.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_neighbors: Number of nearest neighbors to use
        n_components: Target dimensionality
    
    Returns:
        Embedding Y of shape (n_samples, n_components)
    """
    n_samples = X.shape[0]
    
    # Step 1: Find k nearest neighbors for each point
    # Compute pairwise Euclidean distances
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
    
    # Get indices of k nearest neighbors for each point (excluding self)
    neighbor_indices = np.zeros((n_samples, n_neighbors), dtype=int)
    for i in range(n_samples):
        # Sort distances and get indices of nearest neighbors (skip first which is self)
        sorted_indices = np.argsort(distances[i])
        neighbor_indices[i] = sorted_indices[1:n_neighbors+1]
    
    # Step 2: Compute reconstruction weights
    # For each point, solve for weights that reconstruct it from its neighbors
    W = np.zeros((n_samples, n_samples))
    reg = 1e-3  # regularization term
    
    for i in range(n_samples):
        # Get neighbors of point i
        neighbors = neighbor_indices[i]
        k = n_neighbors
        
        # Compute the local covariance matrix
        # Z = X[i] - X[neighbors]
        Z = X[neighbors] - X[i]  # Shape: (k, n_features)
        
        # Compute Gram matrix G = Z @ Z.T
        G = Z @ Z.T
        G += reg * np.eye(k)  # Add regularization
        
        # Solve G @ w = 1 for weights
        # We want weights that sum to 1
        ones = np.ones(k)
        try:
            w = np.linalg.solve(G, ones)
            w = w / np.sum(w)  # Normalize to sum to 1
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            w = np.linalg.lstsq(G, ones, rcond=None)[0]
            w = w / np.sum(w)
        
        W[i, neighbors] = w
    
    # Step 3: Compute the embedding
    # Solve for Y that minimizes sum_i ||Y[i] - sum_j W[i,j] Y[j]||^2
    # This is equivalent to solving (I - W)^T (I - W) Y = lambda Y
    # and taking the eigenvectors corresponding to the smallest eigenvalues
    # (excluding the zero eigenvalue)
    
    M = (np.eye(n_samples) - W).T @ (np.eye(n_samples) - W)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # Sort by eigenvalues (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Skip the first eigenvector (smallest eigenvalue, which is approximately 0)
    # and take the next n_components eigenvectors
    Y = eigenvectors[:, 1:n_components+1]
    
    # Standardize the sign of each embedding dimension
    # Make sum of elements in each column positive
    for col in range(Y.shape[1]):
        if np.sum(Y[:, col]) < 0:
            Y[:, col] = -Y[:, col]
    
    return Y