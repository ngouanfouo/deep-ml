def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of gradient descent training.
    """
    import torch.nn.functional as F
    
    # Step 1: Zero gradients (manual approach works everywhere)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # Step 2: Forward pass
    outputs = model(x_batch)
    
    # Step 3: Compute loss
    loss = F.cross_entropy(outputs, y_batch)
    
    # Step 4: Backward pass
    loss.backward()
    
    # Step 5: Update parameters
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad
    
    # Step 6: Return loss as float
    return loss.item()