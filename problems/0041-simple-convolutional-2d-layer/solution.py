import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """
    Perform a 2D convolution operation on an input matrix.
    
    Args:
        input_matrix: 2D numpy array of shape (input_height, input_width)
        kernel: 2D numpy array of shape (kernel_height, kernel_width)
        padding: Integer indicating zero-padding size applied to all borders
        stride: Integer indicating spatial step size of the window slide
        
    Returns:
        2D numpy array containing the convoluted outputs.
    """
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 1. Apply zero-padding symmetrically to the margins of the input matrix
    # np.pad adds 'padding' rows/columns of zeros to the top, bottom, left, and right
    padded_matrix = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    
    # Update dimensions based on the new padded layout
    padded_height, padded_width = padded_matrix.shape

    # 2. Calculate the exact output grid dimensions using the standard formula
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    # Initialize a blank destination matrix filled with float zeros
    output_matrix = np.zeros((output_height, output_width), dtype=float)

    # 3. Perform the sliding window operation (convolution / element-wise dot product)
    for i in range(output_height):
        for j in range(output_width):
            # Calculate the spatial bounding coordinates for the current window on the padded input
            start_row = i * stride
            end_row = start_row + kernel_height
            
            start_col = j * stride
            end_col = start_col + kernel_width
            
            # Slice out the local region of interest (receptive field)
            receptive_field = padded_matrix[start_row:end_row, start_col:end_col]
            
            # Compute element-wise multiplication and sum the results
            output_matrix[i, j] = np.sum(receptive_field * kernel)
            
    return output_matrix

