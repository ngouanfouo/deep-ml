import numpy as np

def _mean_blur(frame, k):
    """Mean blur with edge padding, applied per channel (channel axis untouched)."""
    H, W, C = frame.shape
    pad = k // 2
    # Pad only the spatial axes with edge replication; channels unpadded
    padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    out = np.zeros_like(frame, dtype=float)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + H, dx:dx + W, :]
    return out / (k * k)


def frame_aware_corrupt(
    frames: np.ndarray,
    gaussian_prob: float,
    gaussian_std: float,
    color_shift_prob: float,
    color_shift_range: float,
    blur_prob: float,
    blur_kernel_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply per-frame stochastic corruption to simulate history drift.

    Args:
        frames:            float array of shape (T, H, W, C)
        gaussian_prob:     probability of Gaussian noise per frame
        gaussian_std:      std of Gaussian noise
        color_shift_prob:  probability of color shift per frame
        color_shift_range: uniform shift range per channel
        blur_prob:         probability of mean blur per frame
        blur_kernel_size:  square kernel side length for blur
        rng:               seeded numpy random generator

    Returns:
        Corrupted array of shape (T, H, W, C), dtype float
    """
    frames = np.asarray(frames, dtype=float)
    T, H, W, C = frames.shape
    out = frames.copy()
    
    for t in range(T):
        frame = out[t]  # view into out; reassigned to new arrays as we go
        
        # 1. Additive Gaussian noise
        if rng.random() < gaussian_prob:
            frame = frame + rng.normal(0.0, gaussian_std, size=frame.shape)
        
        # 2. Per-channel color shift (same shift added to every spatial location)
        if rng.random() < color_shift_prob:
            shift = rng.uniform(-color_shift_range, color_shift_range, size=C)
            frame = frame + shift  # broadcasts over H, W
        
        # 3. Mean blur with edge padding
        if rng.random() < blur_prob:
            frame = _mean_blur(frame, blur_kernel_size)
        
        out[t] = frame
    
    return out