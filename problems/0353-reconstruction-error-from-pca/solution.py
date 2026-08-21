import torch

def pca_reconstruction_error(X: torch.Tensor, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features) as a torch.Tensor
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_samples, n_features = X.shape
    
    # Center the data (subtract mean of each feature)
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean
    
    # Compute covariance matrix using population covariance (divide by n, not n-1)
    covariance_matrix = (X_centered.T @ X_centered) / n_samples
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Project data onto principal components (reduce dimensionality)
    # Z = X_centered @ components
    projected = X_centered @ components
    
    # Reconstruct data by projecting back to original space
    X_reconstructed = projected @ components.T + mean
    
    # Calculate Mean Squared Error between original and reconstructed
    mse = torch.mean((X - X_reconstructed) ** 2)
    
    # Return as float
    return mse.item()