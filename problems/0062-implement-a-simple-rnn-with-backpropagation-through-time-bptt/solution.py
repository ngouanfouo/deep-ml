import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases using standard scaling.
        """
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Enforce the exact test suite initialization parameters
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Internal tracking caches
        self.xs = {}
        self.hs = {}
        self.ys = {}

    def forward(self, x):
        """
        Forward pass through the RNN preserving explicit 3D tensor configurations.
        """
        T = len(x)
        self.hs[-1] = np.zeros((self.hidden_size, 1))
        
        outputs = []
        for t in range(T):
            # Isolate input scalar/vector as a clean column matrix
            self.xs[t] = np.array(x[t], dtype=float).reshape(-1, 1)
            
            # Recurrent transformation equation: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            raw_hidden = np.dot(self.W_xh, self.xs[t]) + np.dot(self.W_hh, self.hs[t-1]) + self.b_h
            self.hs[t] = np.tanh(raw_hidden)
            
            # Linear output projection equation: y_t = W_hy * h_t + b_y
            self.ys[t] = np.dot(self.W_hy, self.hs[t]) + self.b_y
            
            # Return matching the exact expected shape layout: [output_size, 1]
            outputs.append(self.ys[t].copy())
            
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time (BPTT) with explicit derivative accumulations.
        """
        T = len(x)
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        # Traversal backwards through the sequence steps
        for t in reversed(range(T)):
            target = np.array(y[t], dtype=float).reshape(-1, 1)
            
            # Output error gradient evaluation
            dy = self.ys[t] - target
            
            # Weight update calculations for outer projection matrix
            dW_hy += np.dot(dy, self.hs[t].T)
            db_y += dy
            
            # Hidden state derivative evaluation hook
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1.0 - self.hs[t] ** 2) * dh
            
            # Parameter structural updates accumulation
            dW_xh += np.dot(dh_raw, self.xs[t].T)
            dW_hh += np.dot(dh_raw, self.hs[t-1].T)
            db_h += dh_raw
            
            # Route state dynamics gradient to step t-1
            dh_next = np.dot(self.W_hh.T, dh_raw)
            
        # Optimization parameter changes execution step
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h  -= learning_rate * db_h
        self.b_y  -= learning_rate * db_y

