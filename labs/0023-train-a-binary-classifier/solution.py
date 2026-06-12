import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier using logistic regression from scratch.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    
    # Add bias term (column of ones) to X
    X_train_with_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_val_with_bias = np.column_stack([np.ones(X_val.shape[0]), X_val])
    
    # Initialize weights
    n_features = X_train_with_bias.shape[1]
    weights = np.zeros(n_features)
    
    # Hyperparameters
    learning_rate = 0.01
    n_iterations = 1000
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Gradient descent training
    for i in range(n_iterations):
        # Forward pass
        z = np.dot(X_train_with_bias, weights)
        predictions = sigmoid(z)
        
        # Compute gradient
        gradient = np.dot(X_train_with_bias.T, (predictions - y_train)) / len(y_train)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Early stopping based on validation accuracy (optional)
        if i % 100 == 0:
            val_predictions = sigmoid(np.dot(X_val_with_bias, weights))
            val_pred_binary = (val_predictions >= 0.5).astype(int)
            val_accuracy = np.mean(val_pred_binary == y_val)
            if val_accuracy > 0.98:  # Stop if validation accuracy is very high
                break
    
    # Define the predict function
    def predict(X):
        """
        Predict class labels for input samples.
        
        Args:
            X: numpy array of shape (n_samples, 30) -- standardized features
        
        Returns:
            y_pred: numpy array of shape (n_samples,) with values 0 or 1
        """
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Compute probabilities
        probabilities = sigmoid(np.dot(X_with_bias, weights))
        
        # Convert to binary predictions (threshold at 0.5)
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions
    
    return predict