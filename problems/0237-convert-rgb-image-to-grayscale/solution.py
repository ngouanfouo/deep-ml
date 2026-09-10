import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using the luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Validate and convert to a float numpy array
    try:
        arr = np.asarray(image, dtype=float)
    except (ValueError, TypeError):
        return -1
    
    # Must be 3D with exactly 3 color channels
    if arr.ndim != 3 or arr.shape[2] != 3:
        return -1
    
    # Must have non-empty spatial dimensions
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return -1
    
    # All values must be finite
    if not np.all(np.isfinite(arr)):
        return -1
    
    # All pixel values must be within [0, 255]
    if np.any(arr < 0) or np.any(arr > 255):
        return -1
    
    # Luminosity method: Gray = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    
    # Round to nearest integer and return as nested list
    gray_rounded = np.round(gray).astype(int)
    return gray_rounded.tolist()