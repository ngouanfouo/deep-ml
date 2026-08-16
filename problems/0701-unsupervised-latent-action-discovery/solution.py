import numpy as np

def latent_action_forward(obs_t: np.ndarray,
                          obs_tp1: np.ndarray,
                          W_enc: np.ndarray,
                          b_enc: np.ndarray,
                          codebook: np.ndarray,
                          W_dec: np.ndarray,
                          b_dec: np.ndarray) -> np.ndarray:
    """
    Forward pass for unsupervised latent action discovery.

    Encodes (o_t, o_{t+1}) pairs into a continuous latent, snaps the latent
    to the nearest codebook vector (the discovered discrete action), and
    decodes the predicted next observation from (o_t, z_q).

    Returns:
        Predicted next observations of shape (B, D_obs).
    """
    # 1. Inverse-dynamics encoder: concatenate [obs_t, obs_tp1]
    obs_pair = np.concatenate([obs_t, obs_tp1], axis=1)  # Shape: (B, 2 * D_obs)
    z_e = obs_pair @ W_enc + b_enc                        # Shape: (B, D_latent)

    # 2. Discrete bottleneck: find nearest codebook entry in squared Euclidean distance
    # Compute squared distances between each latent vector and each codebook entry
    # Shape of dists: (B, K)
    dists = np.sum((z_e[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=-1)
    
    # np.argmin automatically breaks ties by selecting the lowest index
    indices = np.argmin(dists, axis=1)                   # Shape: (B,)
    z_q = codebook[indices]                               # Shape: (B, D_latent)

    # 3. Forward-dynamics decoder: concatenate [obs_t, z_q]
    dec_in = np.concatenate([obs_t, z_q], axis=1)        # Shape: (B, D_obs + D_latent)
    obs_tp1_pred = dec_in @ W_dec + b_dec                 # Shape: (B, D_obs)

    return obs_tp1_pred