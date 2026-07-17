import torch

def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    # 1. Handle edge case: Empty image outer structure
    if not img or not isinstance(img, list):
        return -1
        
    first_row_len = len(img[0])
    # 2. Handle edge case: Empty inner row matrix structure
    if first_row_len == 0:
        return -1
        
    total_sum = 0.0
    total_pixels = 0
    
    # 3. Handle structural and range properties row by row
    for row in img:
        # Inconsistent row length checking
        if not isinstance(row, list) or len(row) != first_row_len:
            return -1
            
        # Convert row to a PyTorch tensor to perform vectorized checks
        row_tensor = torch.tensor(row, dtype=torch.float32)
        
        # Out of bounds pixel value checking (0-255)
        if torch.any(row_tensor < 0) or torch.any(row_tensor > 255):
            return -1
            
        total_sum += torch.sum(row_tensor).item()
        total_pixels += len(row)
        
    # Calculate global average brightness
    average_brightness = total_sum / total_pixels
    
    return round(float(average_brightness), 2)