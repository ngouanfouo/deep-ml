import torch

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.
    
    Args:
        image: 2D list/array or torch.Tensor representing a grayscale image
               with values in range [0, 255]
    
    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    try:
        # Convert input to tensor
        if isinstance(image, list):
            # Check if it's a valid 2D list
            if not image:
                return -1
            if not all(isinstance(row, list) for row in image):
                return -1
            row_lengths = [len(row) for row in image]
            if len(set(row_lengths)) != 1 or any(length == 0 for length in row_lengths):
                return -1
            image_tensor = torch.tensor(image, dtype=torch.float32)
        elif isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            return -1
        
        # Check dimensions
        if image_tensor.dim() != 2:
            return -1
        
        height, width = image_tensor.shape
        
        # Check minimum size requirement (3x3)
        if height < 3 or width < 3:
            return -1
        
        # Check pixel values are in range [0, 255]
        if torch.any(image_tensor < 0) or torch.any(image_tensor > 255):
            return -1
        
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 1.0]], dtype=torch.float32)
        
        # Manual convolution for Sobel operator
        # Output size: (height-2, width-2)
        output_height = height - 2
        output_width = width - 2
        
        grad_x = torch.zeros((output_height, output_width), dtype=torch.float32)
        grad_y = torch.zeros((output_height, output_width), dtype=torch.float32)
        
        # Apply Sobel kernels manually
        for i in range(output_height):
            for j in range(output_width):
                # Extract 3x3 patch
                patch = image_tensor[i:i+3, j:j+3]
                # Apply Gx kernel
                grad_x[i, j] = torch.sum(patch * sobel_x)
                # Apply Gy kernel
                grad_y[i, j] = torch.sum(patch * sobel_y)
        
        # Compute gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to [0, 255]
        max_val = torch.max(magnitude)
        
        if max_val > 0:
            normalized = (magnitude / max_val) * 255
        else:
            normalized = magnitude
        
        # Round to nearest integer and convert to list of lists
        result = torch.round(normalized).int().tolist()
        
        return result
        
    except Exception as e:
        # If any error occurs, return -1
        return -1