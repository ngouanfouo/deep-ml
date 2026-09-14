import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    results = {}
    
    # 1. Sigmoid derivative using autograd
    x_sig = torch.tensor(float(x), requires_grad=True)
    out_sig = torch.sigmoid(x_sig)
    out_sig.backward()
    results['sigmoid'] = x_sig.grad.item()
    
    # 2. Tanh derivative using autograd
    x_tanh = torch.tensor(float(x), requires_grad=True)
    out_tanh = torch.tanh(x_tanh)
    out_tanh.backward()
    results['tanh'] = x_tanh.grad.item()
    
    # 3. ReLU derivative using autograd
    x_relu = torch.tensor(float(x), requires_grad=True)
    out_relu = torch.relu(x_relu)
    out_relu.backward()
    results['relu'] = x_relu.grad.item()
    
    return results