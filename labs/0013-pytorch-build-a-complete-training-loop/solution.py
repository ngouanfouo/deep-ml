import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
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
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    """
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Number of training samples
    n_train = X_train.shape[0]
    
    history = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Shuffle training data
        permutation = torch.randperm(n_train)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        # Mini-batch training
        for i in range(0, n_train, batch_size):
            # Get batch
            batch_end = min(i + batch_size, n_train)
            X_batch = X_train_shuffled[i:batch_end]
            y_batch = y_train_shuffled[i:batch_end]
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Process validation data in batches (for efficiency)
            n_val = X_val.shape[0]
            for i in range(0, n_val, batch_size):
                batch_end = min(i + batch_size, n_val)
                X_batch = X_val[i:batch_end]
                y_batch = y_val[i:batch_end]
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Accumulate validation loss
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / ((n_val + batch_size - 1) // batch_size)
        val_accuracy = 100.0 * correct / total
        
        # Store metrics
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
    
    return history