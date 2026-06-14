import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        """
        # Add a small deterministic offset
        # This shifts all pixel values by a constant, maintaining shape
        offset = 0.1
        x_transformed = x + offset
        
        # Clip to [0, 1] to keep values in valid range
        x_transformed = x_transformed.clamp(0.0, 1.0)
        
        return x_transformed