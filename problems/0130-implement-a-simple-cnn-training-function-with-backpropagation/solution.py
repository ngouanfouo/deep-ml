import numpy as np

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.
    '''
    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    np.random.seed(42)  # Ensure reproducibility
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    # Training loop
    for epoch in range(epochs):
        # Process each sample
        for i in range(n_samples):
            # ---------- Forward pass ----------
            # Input image
            x = X[i]
            
            # 1. Convolutional layer with ReLU
            conv_output = np.zeros((output_height, output_width, num_filters))
            for h in range(output_height):
                for w in range(output_width):
                    region = x[h:h+kernel_size, w:w+kernel_size]
                    for f in range(num_filters):
                        conv_output[h, w, f] = np.sum(region * W_conv[:, :, f]) + b_conv[f]
            
            # ReLU activation
            relu_output = np.maximum(0, conv_output)
            
            # 2. Flatten
            flattened = relu_output.flatten()
            
            # 3. Dense layer (logits)
            logits = flattened @ W_dense + b_dense
            
            # 4. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # 5. Cross entropy loss
            loss = -np.sum(y[i] * np.log(probs + 1e-8))
            
            # ---------- Backward pass ----------
            # Gradient of loss w.r.t. logits (softmax + cross entropy)
            dlogits = probs - y[i]
            
            # Gradient for dense layer
            dW_dense = np.outer(flattened, dlogits)
            db_dense = dlogits
            
            # Gradient w.r.t. flattened input
            dflattened = dlogits @ W_dense.T
            
            # Reshape to relu output shape
            drelu = dflattened.reshape(output_height, output_width, num_filters)
            
            # Gradient through ReLU (drelu = dout where relu > 0, else 0)
            dconv = drelu * (relu_output > 0)
            
            # Gradient for convolutional layer
            dW_conv = np.zeros_like(W_conv)
            db_conv = np.zeros_like(b_conv)
            
            for h in range(output_height):
                for w in range(output_width):
                    region = x[h:h+kernel_size, w:w+kernel_size]
                    for f in range(num_filters):
                        dW_conv[:, :, f] += dconv[h, w, f] * region
                        db_conv[f] += dconv[h, w, f]
            
            # Update weights and biases for this sample (SGD, not batch GD)
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv
            W_dense -= learning_rate * dW_dense
            b_dense -= learning_rate * db_dense
    
    return W_conv, b_conv, W_dense, b_dense