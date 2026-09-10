import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Validate direction
    if direction not in ('horizontal', 'vertical'):
        return -1
    
    # Validate and convert input
    try:
        arr = np.asarray(image)
    except (ValueError, TypeError):
        return -1
    
    # Must be 2D (grayscale) or 3D (color)
    if arr.ndim not in (2, 3):
        return -1
    
    # Non-empty spatial dimensions
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return -1
    
    if direction == 'horizontal':
        # Reverse each row (left-right)
        flipped = arr[:, ::-1]
    else:  # vertical
        # Reverse the order of rows (top-bottom)
        flipped = arr[::-1, :]
    
    return flipped.tolist()