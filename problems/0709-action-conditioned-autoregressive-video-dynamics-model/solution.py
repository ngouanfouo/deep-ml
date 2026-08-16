import numpy as np

def rollout_video_dynamics(z0: np.ndarray, actions: np.ndarray,
                           W_scale: np.ndarray, b_scale: np.ndarray,
                           W_shift: np.ndarray, b_shift: np.ndarray,
                           W_hidden: np.ndarray, b_hidden: np.ndarray,
                           W_out: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    """
    Autoregressively roll out a video dynamics model conditioned on actions.

    Args:
        z0: Initial latent, shape (D,)
        actions: Action sequence, shape (T, A)
        W_scale, b_scale: Linear projection to per-feature scale (shape (A, D), (D,))
        W_shift, b_shift: Linear projection to per-feature shift (shape (A, D), (D,))
        W_hidden, b_hidden: Hidden transform weights (shape (D, D), (D,))
        W_out, b_out: Output transform weights (shape (D, D), (D,))

    Returns:
        Predicted latent sequence of shape (T, D)
    """
    T = actions.shape[0]
    D = z0.shape[0]
    
    latent_sequence = np.zeros((T, D))
    z_current = z0.copy()

    for t in range(T):
        action = actions[t]
        
        # 1. Compute per-feature scale and shift parameters from action
        gamma = action @ W_scale + b_scale  # Shape: (D,)
        beta = action @ W_shift + b_shift   # Shape: (D,)
        
        # 2. Modulate current latent vector
        z_mod = gamma * z_current + beta     # Shape: (D,)
        
        # 3. Apply hidden layer with tanh activation
        h = np.tanh(z_mod @ W_hidden + b_hidden)  # Shape: (D,)
        
        # 4. Output projection
        delta_z = h @ W_out + b_out          # Shape: (D,)
        
        # 5. Residual update to predict next latent vector
        z_current = z_current + delta_z
        
        # Store prediction
        latent_sequence[t] = z_current

    return latent_sequence