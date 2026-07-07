import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with training and validation metrics.
    """
    # 1. Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    num_samples = X_train.size(0)
    
    # 3. Loop over epochs
    for epoch in range(1, epochs + 1):
        # Set model to training mode
        model.train()
        
        # a. Shuffle training data indices
        permutation = torch.randperm(num_samples)
        
        running_loss = 0.0
        
        # b. Loop over mini-batches
        for i in range(0, num_samples, batch_size):
            # Clear gradients
            optimizer.zero_grad()
            
            # Extract mini-batch indices and data
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss scaled by the actual batch size
            running_loss += loss.item() * batch_X.size(0)
            
        # Calculate average training loss for the entire epoch
        epoch_train_loss = running_loss / num_samples
        
        # c. Compute validation accuracy and loss
        model.eval()
        with torch.no_grad():
            # Forward pass on validation data
            val_outputs = model(X_val)
            epoch_val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate accuracy
            preds = val_outputs.argmax(dim=1)
            correct = (preds == y_val).sum().item()
            epoch_val_accuracy = correct / X_val.size(0)
            
        # d. Append metrics to history
        history.append({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'val_accuracy': epoch_val_accuracy
        })
        
    # 4. Return history
    return history