from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of gradient descent training.
    """
    # Get all model parameters
    params = get_parameters(model)
    
    # 1. Zero gradients: set p.grad = None for each parameter
    for p in params:
        # Ensure requires_grad is True
        if not p.requires_grad:
            p.requires_grad = True
        # Zero out gradients
        p.grad = None
    
    # 2. Forward pass: logits = model(x_batch)
    logits = model(x_batch)
    
    # 3. Compute loss: logits.sparse_categorical_crossentropy(y_batch)
    loss = logits.sparse_categorical_crossentropy(y_batch)
    
    # 4. Backward pass: loss.backward()
    loss.backward()
    
    # 5. Update each parameter manually using p.assign(p.detach() - lr * p.grad)
    for p in params:
        if p.grad is not None:
            p.assign(p.detach() - lr * p.grad)
    
    # 6. Realize the updates and return loss.item()
    Tensor.realize(*params)
    return loss.item()