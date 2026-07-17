def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    if not dense_matrix or not dense_matrix[0]:
        return [], [], [0]

    num_rows = len(dense_matrix)
    num_cols = len(dense_matrix[0])
    
    values = []
    row_indices = []
    column_pointer = [0]
    
    current_non_zero_count = 0
    
    # Traverse column by column (Column-Major Order)
    for c in range(num_cols):
        for r in range(num_rows):
            element = dense_matrix[r][c]
            if element != 0:
                values.append(element)
                row_indices.append(r)
                current_non_zero_count += 1
        
        # Append the starting point index for the next column
        column_pointer.append(current_non_zero_count)
        
    return values, row_indices, column_pointer