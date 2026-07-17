import torch

def calculate_contrast(img: torch.Tensor) -> float:
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (torch.Tensor): 2D tensor representing a grayscale image with pixel values between 0 and 255.
    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    # Handle the edge case of an empty tensor or invalid dimension
    if img.numel() == 0:
        return 0.0
        
    # Calculate the maximum and minimum values safely across the whole tensor structure
    max_val = torch.max(img)
    min_val = torch.min(img)
    
    # Compute contrast difference
    contrast = max_val - min_val
    
    # Cast out to raw native Python float and round to 3 decimal places
    return round(float(contrast.item()), 3)