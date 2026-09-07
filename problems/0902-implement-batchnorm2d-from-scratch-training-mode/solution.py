import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    """
    Performs training-mode batch normalization for a 4-D input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        gamma (torch.Tensor): Scale tensor of shape (C,).
        beta (torch.Tensor): Shift tensor of shape (C,).
        eps (float): Small value added to variance for numerical stability.

    Returns:
        torch.Tensor: Normalized, scaled, and shifted tensor of shape (N, C, H, W).
    """
    # Compute mean and variance over the batch (0) and spatial (2, 3) dimensions
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    # Use population (biased) variance formula as specified
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    # Normalize
    x_hat = (x - mean) / torch.sqrt(var + eps)

    # Reshape gamma and beta to broadcast against (N, C, H, W) -> shape (1, C, 1, 1)
    gamma_reshaped = gamma.view(1, -1, 1, 1)
    beta_reshaped = beta.view(1, -1, 1, 1)

    # Scale and shift
    return gamma_reshaped * x_hat + beta_reshaped