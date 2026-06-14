import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
    """
    # Add bias column of ones as the first column
    m, n = X.shape  # m = number of samples, n = number of features
    X_with_bias = np.hstack([np.ones((m, 1)), X])  # shape (m, n+1)
    
    # Determine number of classes
    C = int(np.max(y)) + 1
    
    # Initialize weights to zero (C x (n+1) matrix)
    W = np.zeros((C, n + 1))
    
    # Store loss values
    losses = []
    
    # Gradient descent
    for _ in range(iterations):
        # Compute scores (logits) for all classes
        # scores shape: (m, C)
        scores = X_with_bias @ W.T
        
        # Apply softmax to get probabilities
        # Subtract max for numerical stability
        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_stable)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # shape (m, C)
        
        # Compute sum-based Cross Entropy loss
        # L = -sum_i log(probs_i[y_i])
        log_probs = np.log(probs[np.arange(m), y])
        loss = -np.sum(log_probs)
        losses.append(loss)
        
        # Compute gradient
        # For each sample i, gradient for class j:
        # dW_j = (probs[i][j] - 1_{y_i == j}) * X_i
        grad = np.zeros_like(W)
        for i in range(m):
            for j in range(C):
                indicator = 1.0 if y[i] == j else 0.0
                grad[j] += (probs[i][j] - indicator) * X_with_bias[i]
        
        # Update weights
        W = W - learning_rate * grad
    
    # Convert to list of lists for the output format
    coefficients = W.tolist()
    
    return coefficients, losses