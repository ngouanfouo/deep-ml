import torch
import math

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.
    
    Args:
        X: List of data points (will be converted to torch.Tensor)
        K: Number of mixture components
        initial_means: List of initial means for each component
        initial_variances: List of initial variances for each component
        initial_weights: List of initial mixture weights (should sum to 1)
        n_iterations: Number of EM iterations to run
    
    Returns:
        Dictionary with 'means', 'variances', 'weights' as lists rounded to 4 decimals
    """
    # Convert data to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    N = len(X_tensor)
    
    # Initialize parameters
    means = torch.tensor(initial_means, dtype=torch.float32)
    variances = torch.tensor(initial_variances, dtype=torch.float32)
    weights = torch.tensor(initial_weights, dtype=torch.float32)
    
    # EM algorithm
    for iteration in range(n_iterations):
        # E-step: Compute responsibilities
        # Calculate probability of each point under each component
        probabilities = torch.zeros(N, K)
        for k in range(K):
            # Gaussian PDF: (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))
            prob = (1 / torch.sqrt(2 * math.pi * variances[k])) * \
                   torch.exp(-(X_tensor - means[k])**2 / (2 * variances[k]))
            probabilities[:, k] = weights[k] * prob
        
        # Normalize to get responsibilities (soft assignments)
        # Add small epsilon to avoid division by zero
        responsibilities = probabilities / (torch.sum(probabilities, dim=1, keepdim=True) + 1e-10)
        
        # M-step: Update parameters
        # For each component
        for k in range(K):
            # Total responsibility for component k
            N_k = torch.sum(responsibilities[:, k])
            
            # Update mean: weighted average of data points
            if N_k > 0:
                means[k] = torch.sum(responsibilities[:, k] * X_tensor) / N_k
                
                # Update variance: weighted average of squared deviations
                variances[k] = torch.sum(responsibilities[:, k] * (X_tensor - means[k])**2) / N_k
                
                # Update weight: proportion of data assigned to component
                weights[k] = N_k / N
    
    # Round to 4 decimal places and convert to lists
    means_rounded = [round(m.item(), 4) for m in means]
    variances_rounded = [round(v.item(), 4) for v in variances]
    weights_rounded = [round(w.item(), 4) for w in weights]
    
    return {
        'means': means_rounded,
        'variances': variances_rounded,
        'weights': weights_rounded
    }