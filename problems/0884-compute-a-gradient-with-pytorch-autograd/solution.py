import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    
    # Create a leaf tensor with requires_grad=True to track operations
    x = torch.tensor(x_value, requires_grad=True, dtype=torch.float32)
    
    # Compute f(x) = x^2 + 3x + 2
    f = x**2 + 3*x + 2
    
    # Trigger backpropagation
    f.backward()
    
    # Extract the gradient and convert to Python float
    return x.grad.item()