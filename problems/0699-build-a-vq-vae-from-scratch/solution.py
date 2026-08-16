import numpy as np

def vq_vae_loss(x: np.ndarray, z_e: np.ndarray, x_recon: np.ndarray, codebook: np.ndarray, beta: float) -> float:
    """
    Compute the VQ-VAE training loss.

    Args:
        x: Original inputs, shape (N, input_dim).
        z_e: Encoder outputs, shape (N, D).
        x_recon: Decoder reconstructions, shape (N, input_dim).
        codebook: Codebook embeddings, shape (K, D).
        beta: Commitment loss weight.

    Returns:
        Total VQ-VAE loss as a single float scalar.
    """
    # 1. Compute squared Euclidean distance between each z_e (N, D) and codebook entry (K, D)
    # Using expansion: ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e . e_k
    distances = (
        np.sum(z_e ** 2, axis=1, keepdims=True)
        + np.sum(codebook ** 2, axis=1)
        - 2 * np.dot(z_e, codebook.T)
    )
    
    # 2. Find the nearest codebook index for each z_e row
    encoding_indices = np.argmin(distances, axis=1)
    
    # 3. Construct the quantized representations z_q
    z_q = codebook[encoding_indices]
    
    # 4. Compute individual loss terms using element-wise mean squared error (MSE)
    # Term 1: Reconstruction Loss
    recon_loss = np.mean((x - x_recon) ** 2)
    
    # Term 2: Codebook / VQ Loss (moves codebook vectors toward encoder outputs)
    # sg[z_e] - z_q -> calculated as MSE between z_e (treated as constant) and z_q
    codebook_loss = np.mean((z_e - z_q) ** 2)
    
    # Term 3: Commitment Loss (keeps encoder outputs close to chosen codebook vectors)
    # z_e - sg[z_q] -> calculated as MSE between z_e and z_q (treated as constant)
    commitment_loss = np.mean((z_e - z_q) ** 2)
    
    # 5. Combine total loss
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    
    return float(total_loss)