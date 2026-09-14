import torch

def latent_encode_decode(
    x: torch.Tensor,
    W_enc_mu: torch.Tensor,
    b_enc_mu: torch.Tensor,
    W_enc_logvar: torch.Tensor,
    b_enc_logvar: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
    epsilon: torch.Tensor
) -> dict:
    """
    Perform variational encoding, latent sampling, and decoding
    as used in latent diffusion models.

    Args:
        x: Input tensor of shape (d,)
        W_enc_mu: Encoder mean weights, shape (d, latent_dim)
        b_enc_mu: Encoder mean bias, shape (latent_dim,)
        W_enc_logvar: Encoder log-variance weights, shape (d, latent_dim)
        b_enc_logvar: Encoder log-variance bias, shape (latent_dim,)
        W_dec: Decoder weights, shape (latent_dim, d)
        b_dec: Decoder bias, shape (d,)
        epsilon: Noise sample tensor, shape (latent_dim,)

    Returns:
        dict with keys 'mu', 'log_var', 'z', 'x_recon' (all torch.Tensor)
    """
    # Align all tensors to match W_enc_mu's device and dtype to prevent type mismatch errors
    dtype = W_enc_mu.dtype
    device = W_enc_mu.device
    
    x = x.to(dtype=dtype, device=device)
    b_enc_mu = b_enc_mu.to(dtype=dtype, device=device)
    W_enc_logvar = W_enc_logvar.to(dtype=dtype, device=device)
    b_enc_logvar = b_enc_logvar.to(dtype=dtype, device=device)
    W_dec = W_dec.to(dtype=dtype, device=device)
    b_dec = b_dec.to(dtype=dtype, device=device)
    epsilon = epsilon.to(dtype=dtype, device=device)
    
    # 1. Compute encoder mean (mu)
    mu = torch.matmul(x, W_enc_mu) + b_enc_mu
    
    # 2. Compute encoder log-variance (log_var)
    log_var = torch.matmul(x, W_enc_logvar) + b_enc_logvar
    
    # 3. Compute standard deviation using std = exp(0.5 * log_var)
    std = torch.exp(0.5 * log_var)
    
    # 4. Sample latent vector using the reparameterization trick: z = mu + std * epsilon
    z = mu + std * epsilon
    
    # 5. Decode latent vector back to original space: x_recon = z @ W_dec + b_dec
    x_recon = torch.matmul(z, W_dec) + b_dec
    
    return {
        'mu': mu,
        'log_var': log_var,
        'z': z,
        'x_recon': x_recon
    }