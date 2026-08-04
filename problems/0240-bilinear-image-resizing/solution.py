import torch
import math

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image: 2D (grayscale) or 3D (RGB) tensor/array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image
    
    Returns:
        Resized image as a nested list with values rounded to 2 decimal places
    """
    if isinstance(image, list):
        image = torch.tensor(image, dtype=torch.float32)
    elif not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    if image.dim() == 2:
        height, width = image.shape
        channels = 1
        is_grayscale = True
    else:
        height, width, channels = image.shape
        is_grayscale = False
    
    # Scale factors for mapping output coordinates to source coordinates
    scale_h = height / new_height
    scale_w = width / new_width
    
    if is_grayscale:
        output = torch.zeros((new_height, new_width), dtype=torch.float32)
    else:
        output = torch.zeros((new_height, new_width, channels), dtype=torch.float32)
    
    for out_h in range(new_height):
        # Map output position to source space
        src_h = out_h * scale_h
        
        # Clamp coordinates to ensure valid bounding box inside original dimensions
        h0 = int(math.floor(src_h))
        h0 = max(0, min(h0, height - 1))
        h1 = min(h0 + 1, height - 1)
        
        dh = src_h - h0
        dh = max(0.0, min(1.0, dh))
        
        for out_w in range(new_width):
            src_w = out_w * scale_w
            
            w0 = int(math.floor(src_w))
            w0 = max(0, min(w0, width - 1))
            w1 = min(w0 + 1, width - 1)
            
            dw = src_w - w0
            dw = max(0.0, min(1.0, dw))
            
            if is_grayscale:
                top_left = image[h0, w0]
                top_right = image[h0, w1]
                bottom_left = image[h1, w0]
                bottom_right = image[h1, w1]
                
                value = (1 - dh) * (1 - dw) * top_left + \
                        (1 - dh) * dw * top_right + \
                        dh * (1 - dw) * bottom_left + \
                        dh * dw * bottom_right
                
                output[out_h, out_w] = value
            else:
                for c in range(channels):
                    top_left = image[h0, w0, c]
                    top_right = image[h0, w1, c]
                    bottom_left = image[h1, w0, c]
                    bottom_right = image[h1, w1, c]
                    
                    value = (1 - dh) * (1 - dw) * top_left + \
                            (1 - dh) * dw * top_right + \
                            dh * (1 - dw) * bottom_left + \
                            dh * dw * bottom_right
                    
                    output[out_h, out_w, c] = value
    
    # Convert output tensor to nested list and apply standard rounding to 2 decimal places
    out_list = output.tolist()
    
    def round_nested(lst):
        if isinstance(lst, list):
            return [round_nested(item) for item in lst]
        return round(lst, 2)
    
    return round_nested(out_list)