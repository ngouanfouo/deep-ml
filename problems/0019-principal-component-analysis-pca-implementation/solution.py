import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # 1. Standardize the dataset (mean = 0, std = 1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=0)
    
    # Handle potential division by zero if a feature has zero variance
    std[std == 0] = 1.0
    standardized_data = (data - mean) / std
    
    # 2. Compute the Covariance Matrix
    # Since rows are samples, rowvar=False calculates covariance between columns (features)
    covariance_matrix = np.cov(standardized_data, rowvar=False, ddof=0)
    
    # 3. Find the eigenvalues and eigenvectors using eigh (optimized for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 4. Sort eigenvalues and eigenvectors in descending order
    # np.linalg.eigh returns them in ascending order, so we reverse them
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 5. Select the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # 6. Apply the Sign Convention
    # For each eigenvector, find its first element with abs > 1e-10.
    # If that element is negative, multiply the entire eigenvector by -1.
    for i in range(k):
        ev = top_k_eigenvectors[:, i]
        # Find the index of the first element matching the condition
        valid_indices = np.where(np.abs(ev) > 1e-10)[0]
        if len(valid_indices) > 0:
            first_significant_element = ev[valid_indices[0]]
            if first_significant_element < 0:
                top_k_eigenvectors[:, i] *= -1
                
    # 7. Round to 4 decimal places as requested
    return np.round(top_k_eigenvectors, 4)

