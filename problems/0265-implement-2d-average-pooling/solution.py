def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
    """
    Perform 2D average pooling on the input matrix.
    
    Args:
        input_matrix: 2D input array of shape (H, W)
        pool_size: Size of the square pooling window
        
    Returns:
        2D array after average pooling of shape (H//pool_size, W//pool_size)
    """
    H = len(input_matrix)
    W = len(input_matrix[0])
    
    out_h = H // pool_size
    out_w = W // pool_size
    
    output = []
    
    for i in range(out_h):
        row = []
        for j in range(out_w):
            # Extract and sum elements in the current pooling window
            window_sum = 0.0
            count = pool_size * pool_size
            
            start_r = i * pool_size
            start_c = j * pool_size
            
            for r in range(start_r, start_r + pool_size):
                for c in range(start_c, start_c + pool_size):
                    window_sum += input_matrix[r][c]
            
            # Compute the average and add to the row
            row.append(window_sum / count)
        
        output.append(row)
        
    return output