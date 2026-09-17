import torch

def cross_product(a, b) -> torch.Tensor:
    """
    Compute the cross product of two 3D vectors a and b.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Both inputs must be 3-dimensional vectors.")

    if not a.is_floating_point():
        a = a.float()
    if not b.is_floating_point():
        b = b.float()

    return torch.cross(a, b, dim=0)