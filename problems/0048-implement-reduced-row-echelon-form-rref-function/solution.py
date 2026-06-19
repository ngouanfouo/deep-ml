import numpy as np

def rref(matrix):
    """
    Convert a matrix to Reduced Row Echelon Form (RREF).
    
    Args:
        matrix: numpy array of shape (m, n)
    
    Returns:
        rref_matrix: numpy array in RREF form
    """
    # Make a copy to avoid modifying the original
    A = matrix.astype(float).copy()
    m, n = A.shape
    
    # Initialize pivot row and column
    pivot_row = 0
    
    # Iterate through columns
    for pivot_col in range(n):
        # Find the pivot row (non-zero entry in current column)
        # Search from pivot_row to bottom
        max_row = pivot_row
        for i in range(pivot_row, m):
            if abs(A[i, pivot_col]) > abs(A[max_row, pivot_col]):
                max_row = i
        
        # If no non-zero entry found in this column, continue to next column
        if abs(A[max_row, pivot_col]) < 1e-10:
            continue
        
        # Swap rows to bring pivot to current pivot_row
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        
        # Scale the pivot row to make pivot entry = 1
        pivot = A[pivot_row, pivot_col]
        A[pivot_row, :] = A[pivot_row, :] / pivot
        
        # Eliminate all other entries in this column
        for i in range(m):
            if i != pivot_row:
                factor = A[i, pivot_col]
                A[i, :] = A[i, :] - factor * A[pivot_row, :]
        
        # Move to next pivot row
        pivot_row += 1
        
        # If we've processed all rows, break
        if pivot_row >= m:
            break
    
    # Clean up near-zero values
    A[np.abs(A) < 1e-10] = 0
    
    return A