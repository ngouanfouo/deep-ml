import torch

def vector_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | int:
    # Vectors must have the same length (shape) for element-wise addition.
    if a.shape != b.shape:
        return -1
    return a + b