import torch
from torch.distributions import Normal, kl_divergence as kl_div_fn
import torch.nn.functional as F

def vae_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: torch.Tensor of shape (batch_size, features), original input
        x_reconstructed: torch.Tensor of shape (batch_size, features), reconstructed input
        mu: torch.Tensor of shape (batch_size, latent_dim), latent mean
        log_var: torch.Tensor of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # 1. Compute reconstruction loss: Mean over batch of the sum of squared differences
    recon_loss = torch.mean(torch.sum((x - x_reconstructed) ** 2, dim=-1))
    
    # 2. Compute KL divergence against a standard normal prior:
    # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var), dim=-1) averaged over batch
    kl_loss = -0.5 * torch.mean(torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
    
    # 3. Total loss is the sum of reconstruction loss and KL divergence
    total_loss = recon_loss + kl_loss
    
    return float(total_loss.item()), float(recon_loss.item()), float(kl_loss.item())