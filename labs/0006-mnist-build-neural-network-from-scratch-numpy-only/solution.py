import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        
        # He (Kaiming) initialization for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Cache for intermediate values during forward pass
        self.cache = {}
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # Layer 1: Linear + ReLU
        z1 = X @ self.W1 + self.b1  # (N, hidden_size)
        a1 = np.maximum(0, z1)       # ReLU activation
        
        # Layer 2: Linear + Softmax
        z2 = a1 @ self.W2 + self.b2  # (N, output_size)
        
        # Stable softmax (subtract max for numerical stability)
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        
        # Cache for backward pass
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2}
        
        return probs
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        N = X.shape[0]
        
        # 1. Compute cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-8))
        
        # 2. Gradient of loss w.r.t. softmax output (z2)
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(N), y] = 1
        dz2 = probs - y_one_hot  # (N, output_size)
        
        # 3. Backprop through layer 2 (linear)
        dW2 = self.cache['a1'].T @ dz2 / N
        db2 = np.sum(dz2, axis=0, keepdims=True) / N
        
        # Backprop to hidden layer: dL/da1 = dz2 @ W2^T
        da1 = dz2 @ self.W2.T  # (N, hidden_size)
        
        # 4. Backprop through ReLU
        dz1 = da1 * (self.cache['z1'] > 0)  # (N, hidden_size)
        
        # 5. Backprop through layer 1 (linear)
        dW1 = X.T @ dz1 / N
        db1 = np.sum(dz1, axis=0, keepdims=True) / N
        
        # 6. Update all parameters using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        return loss
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)