import numpy as np
import sympy

def matrix_image(A):
    """
    Finds the basis vectors that span the column space of A,
    extracted as the linearly independent columns of the original matrix.
    
    :param A: NumPy array representing the input matrix
    :return: List of lists containing the independent column vectors
    """
    # Convert to a SymPy Matrix to compute the Reduced Row Echelon Form (RREF)
    sympy_matrix = sympy.Matrix(A)
    
    # rref() returns the reduced matrix and a tuple of pivot column indices
    _, pivot_indices = sympy_matrix.rref()
    
    # Extract the original columns corresponding to these pivot indices
    independent_columns = A[:, pivot_indices]
    
    # Return as a list of lists to match the required output format
    return independent_columns.tolist()

