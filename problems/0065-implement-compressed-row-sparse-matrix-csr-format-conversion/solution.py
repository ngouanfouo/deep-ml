import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    vals = []
    col_idx = []
    row_ptr = [0]  # start with 0 for first row

    for row in dense_matrix:
        count = 0
        for col, val in enumerate(row):
            if val != 0:
                vals.append(val)
                col_idx.append(col)
                count += 1
        row_ptr.append(row_ptr[-1] + count)

    return vals, col_idx, row_ptr