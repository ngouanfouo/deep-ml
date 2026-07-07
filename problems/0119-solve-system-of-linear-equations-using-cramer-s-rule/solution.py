import numpy as np

def cramers_rule(A, b):
    # Your code here
    # Convert inputs to numpy arrays if they aren't already
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Check if A is square
    if A.shape[0] != A.shape[1]:
        return -1
    
    n = A.shape[0]
    
    # Compute determinant of A
    det_A = np.linalg.det(A)
    
    # If determinant is zero (or very close to zero), no unique solution
    if abs(det_A) < 1e-10:
        return -1
    
    # Initialize solution vector
    x = np.zeros(n)
    
    # For each variable, replace the corresponding column with b
    for i in range(n):
        # Create a copy of A
        A_i = A.copy()
        # Replace the i-th column with b
        A_i[:, i] = b
        # Compute determinant of the modified matrix
        det_A_i = np.linalg.det(A_i)
        # Compute x_i using Cramer's Rule
        x[i] = det_A_i / det_A
    
    return x