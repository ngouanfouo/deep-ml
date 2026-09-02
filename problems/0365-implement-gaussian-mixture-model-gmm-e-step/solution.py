import torch

def gmm_e_step(X: torch.Tensor, means: torch.Tensor, variances: torch.Tensor,
               mixing_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Compute the E-step of Gaussian Mixture Model using PyTorch.

    Args:
        X: Data points of shape (n_samples,)
        means: Component means of shape (n_components,)
        variances: Component variances of shape (n_components,)
        mixing_coeffs: Mixing coefficients of shape (n_components,)

    Returns:
        Responsibility matrix of shape (n_samples, n_components)
    """
    # Ensure inputs are float tensors
    X = X.float()
    means = means.float()
    variances = variances.float()
    mixing_coeffs = mixing_coeffs.float()

    # Reshape X to (n_samples, 1) for broadcasting with means/variances of shape (n_components,)
    X_expanded = X.unsqueeze(1) # Shape: (n_samples, 1)

    # 1. Compute univariate Gaussian probability density for each data point and component
    # PDF = (1 / sqrt(2 * pi * variance)) * exp(- (x - mean)^2 / (2 * variance))
    coef = 1.0 / torch.sqrt(2 * torch.pi * variances)
    exponent = -0.5 * ((X_expanded - means) ** 2) / variances
    pdf = coef * torch.exp(exponent) # Shape: (n_samples, n_components)

    # 2. Weight by mixing coefficients (priors): p(x | k) * P(k)
    weighted_probs = pdf * mixing_coeffs # Shape: (n_samples, n_components)

    # 3. Compute responsibilities (posterior probabilities): normalize across components
    total_probs = weighted_probs.sum(dim=1, keepdim=True) # Shape: (n_samples, 1)
    
    # Handle potential division by zero if all probabilities are zero
    responsibilities = weighted_probs / torch.clamp(total_probs, min=1e-12)

    return responsibilities