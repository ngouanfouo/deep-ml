import torch

def flatten(A: torch.Tensor) -> torch.Tensor:
    """Row-major flatten via index arithmetic (no reshape/view/flatten)."""
    # Your code here
    pass
import torch

def flatten(A: torch.Tensor) -> torch.Tensor:
    """Row-major flatten via index arithmetic (no reshape/view/flatten)."""
    # Get the shape of the input matrix
    rows, cols = A.shape
    
    # Create a range of linear indices from 0 to rows*cols - 1
    linear_indices = torch.arange(rows * cols)
    
    # Compute row indices: linear_indices // cols
    row_indices = linear_indices // cols
    
    # Compute column indices: linear_indices % cols
    col_indices = linear_indices % cols
    
    # Use advanced indexing to gather elements
    # A[row_indices, col_indices] gives the flattened array in row-major order
    result = A[row_indices, col_indices]
    
    return result