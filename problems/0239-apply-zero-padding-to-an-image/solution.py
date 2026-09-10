import torch

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or torch.Tensor of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # --- validate pad_width ---
    if not isinstance(pad_width, int) or isinstance(pad_width, bool):
        return -1
    if pad_width < 0:
        return -1

    # --- convert img to a 2D Python list ---
    if torch.is_tensor(img):
        if img.dim() != 2:
            return -1
        if img.shape[0] == 0 or img.shape[1] == 0:
            return -1
        img_list = img.detach().cpu().tolist()
    elif isinstance(img, (list, tuple)):
        if len(img) == 0:
            return -1
        if not all(isinstance(row, (list, tuple)) for row in img):
            return -1
        W = len(img[0])
        if W == 0:
            return -1
        if any(len(row) != W for row in img):
            return -1
        img_list = [list(row) for row in img]
    elif hasattr(img, 'shape') and hasattr(img, 'tolist'):
        # numpy array or similar
        if len(img.shape) != 2:
            return -1
        if img.shape[0] == 0 or img.shape[1] == 0:
            return -1
        img_list = img.tolist()
    else:
        return -1

    H = len(img_list)
    W = len(img_list[0])

    # ensure integer pixel values
    img_list = [[int(val) for val in row] for row in img_list]

    new_W = W + 2 * pad_width
    zero_row = [0] * new_W

    padded = []
    # top padding
    for _ in range(pad_width):
        padded.append(zero_row.copy())
    # original image with left/right padding
    for row in img_list:
        padded.append([0] * pad_width + row + [0] * pad_width)
    # bottom padding
    for _ in range(pad_width):
        padded.append(zero_row.copy())

    return padded