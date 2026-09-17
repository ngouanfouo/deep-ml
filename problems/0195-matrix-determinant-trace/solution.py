import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Ensure the input is a floating-point tensor for linalg.det
    if not matrix.is_floating_point():
        matrix = matrix.float()

    det = torch.linalg.det(matrix)
    trace = torch.trace(matrix)
    return det, trace