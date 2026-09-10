import numpy as np

def sample_and_preprocess(video, num_frames: int) -> list:
    """
    Uniformly sample num_frames frames from a video and normalize to [0, 1].

    Args:
        video: array-like of shape (T, H, W, C) with values in [0, 255].
        num_frames: number of frames to sample.

    Returns:
        Nested list of shape (num_frames, H, W, C) with float values in [0, 1].
    """
    video = np.array(video, dtype=float)
    T = video.shape[0]
    
    if num_frames == 1:
        indices = np.array([0])
    else:
        # Evenly spaced indices from 0 to T-1 (inclusive), rounded to nearest int
        indices = np.round(np.linspace(0, T - 1, num_frames)).astype(int)
    
    sampled = video[indices]          # shape (num_frames, H, W, C)
    normalized = sampled / 255.0      # scale to [0.0, 1.0]
    
    return normalized.tolist()