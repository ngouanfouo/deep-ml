import numpy as np

def first_frame_anchor_inject(
    history_frames: np.ndarray,
    first_frame: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Prepend a lightly noised first frame anchor to the history.

    Args:
        history_frames: array of shape (T, H, W, C)
        first_frame:    array of shape (H, W, C)
        sigma_min:      minimum noise std
        sigma_max:      maximum noise std
        rng:            seeded numpy random generator

    Returns:
        Array of shape (T+1, H, W, C) where index 0 is the noisy anchor
        and indices 1..T are the original history frames.
    """
    # 1) Sample a single noise level uniformly from [sigma_min, sigma_max].
    sigma = rng.uniform(sigma_min, sigma_max)

    # 2) Add Gaussian noise with that std to the first frame.
    noise = rng.standard_normal(first_frame.shape) * sigma
    noisy_anchor = first_frame + noise

    # 3) Prepend the noisy anchor to the history.
    result = np.concatenate([noisy_anchor[None, ...], history_frames], axis=0)

    return result