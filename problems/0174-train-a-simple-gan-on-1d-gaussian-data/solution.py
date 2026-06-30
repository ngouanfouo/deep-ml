import numpy as np

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, 
              learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, 
              seed: int = 42):
    """
    Train a simple GAN to learn a 1D Gaussian distribution.

    Args:
        mean_real: Mean of the target Gaussian
        std_real: Std of the target Gaussian
        latent_dim: Dimension of the noise input to the generator
        hidden_dim: Hidden layer size for both networks
        learning_rate: Learning rate for gradient descent
        epochs: Number of training epochs
        batch_size: Training batch size
        seed: Random seed for reproducibility

    Returns:
        gen_forward: A function that takes z and returns generated samples
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize weights and biases for generator
    # G: z -> hidden -> output
    g_W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
    g_b1 = np.zeros((1, hidden_dim))
    g_W2 = np.random.randn(hidden_dim, 1) * 0.01
    g_b2 = np.zeros((1, 1))
    
    # Initialize weights and biases for discriminator
    # D: x -> hidden -> sigmoid
    d_W1 = np.random.randn(1, hidden_dim) * 0.01
    d_b1 = np.zeros((1, hidden_dim))
    d_W2 = np.random.randn(hidden_dim, 1) * 0.01
    d_b2 = np.zeros((1, 1))
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    def generator_forward(z):
        # Forward pass through generator
        hidden = relu(np.dot(z, g_W1) + g_b1)
        output = np.dot(hidden, g_W2) + g_b2
        return output, hidden
    
    def discriminator_forward(x):
        # Forward pass through discriminator
        hidden = relu(np.dot(x, d_W1) + d_b1)
        output = sigmoid(np.dot(hidden, d_W2) + d_b2)
        return output, hidden
    
    # Training loop
    for epoch in range(epochs):
        # Generate real samples from target distribution
        real_samples = np.random.normal(mean_real, std_real, (batch_size, 1))
        
        # Generate fake samples
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples, g_hidden = generator_forward(z)
        
        # ----- Train Discriminator -----
        # Forward pass for real samples
        d_real_output, d_real_hidden = discriminator_forward(real_samples)
        
        # Forward pass for fake samples
        d_fake_output, d_fake_hidden = discriminator_forward(fake_samples)
        
        # Discriminator loss: binary cross-entropy
        # L_D = -[log(D(real)) + log(1 - D(fake))]
        d_loss_real = -np.log(d_real_output + 1e-8)
        d_loss_fake = -np.log(1 - d_fake_output + 1e-8)
        d_loss = np.mean(d_loss_real + d_loss_fake)
        
        # Backpropagation for discriminator
        # Gradient for output layer
        d_grad_output_real = -(1 / (d_real_output + 1e-8)) * sigmoid_derivative(np.dot(d_real_hidden, d_W2) + d_b2)
        d_grad_output_fake = (1 / (1 - d_fake_output + 1e-8)) * sigmoid_derivative(np.dot(d_fake_hidden, d_W2) + d_b2)
        
        # Average gradients over batch
        d_grad_output = np.mean(d_grad_output_real + d_grad_output_fake, axis=0, keepdims=True)
        
        # Gradients for W2 and b2
        d_grad_W2 = np.dot(d_real_hidden.T, d_grad_output_real) + np.dot(d_fake_hidden.T, d_grad_output_fake)
        d_grad_W2 /= batch_size
        d_grad_b2 = np.mean(d_grad_output_real + d_grad_output_fake, axis=0, keepdims=True)
        
        # Gradients for hidden layer
        d_grad_hidden_real = np.dot(d_grad_output_real, d_W2.T) * relu_derivative(np.dot(real_samples, d_W1) + d_b1)
        d_grad_hidden_fake = np.dot(d_grad_output_fake, d_W2.T) * relu_derivative(np.dot(fake_samples, d_W1) + d_b1)
        
        # Gradients for W1 and b1
        d_grad_W1 = np.dot(real_samples.T, d_grad_hidden_real) + np.dot(fake_samples.T, d_grad_hidden_fake)
        d_grad_W1 /= batch_size
        d_grad_b1 = np.mean(d_grad_hidden_real + d_grad_hidden_fake, axis=0, keepdims=True)
        
        # Update discriminator parameters
        d_W1 -= learning_rate * d_grad_W1
        d_b1 -= learning_rate * d_grad_b1
        d_W2 -= learning_rate * d_grad_W2
        d_b2 -= learning_rate * d_grad_b2
        
        # ----- Train Generator (non-saturating loss) -----
        # Generate new fake samples
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples, g_hidden = generator_forward(z)
        
        # Discriminator output on fake samples
        d_fake_output, d_fake_hidden = discriminator_forward(fake_samples)
        
        # Generator loss: -log(D(fake))
        g_loss = -np.mean(np.log(d_fake_output + 1e-8))
        
        # Backpropagation for generator
        # Gradient from discriminator output
        g_grad_output = -(1 / (d_fake_output + 1e-8)) * sigmoid_derivative(np.dot(d_fake_hidden, d_W2) + d_b2)
        
        # Gradient through discriminator's first layer to generator's output
        # We need to backprop through the discriminator to get gradients for generator
        g_grad_hidden = np.dot(g_grad_output, d_W2.T) * relu_derivative(np.dot(fake_samples, d_W1) + d_b1)
        
        # Gradient with respect to generator output (which is fake_samples)
        g_grad_input = np.dot(g_grad_hidden, d_W1.T)
        
        # Backprop through generator
        # Gradient for generator output layer
        g_grad_output_layer = g_grad_input * 1  # Linear activation, derivative is 1
        
        # Gradients for W2 and b2
        g_grad_W2 = np.dot(g_hidden.T, g_grad_output_layer)
        g_grad_W2 /= batch_size
        g_grad_b2 = np.mean(g_grad_output_layer, axis=0, keepdims=True)
        
        # Gradients for generator hidden layer
        g_grad_hidden_layer = np.dot(g_grad_output_layer, g_W2.T) * relu_derivative(np.dot(z, g_W1) + g_b1)
        
        # Gradients for W1 and b1
        g_grad_W1 = np.dot(z.T, g_grad_hidden_layer)
        g_grad_W1 /= batch_size
        g_grad_b1 = np.mean(g_grad_hidden_layer, axis=0, keepdims=True)
        
        # Update generator parameters
        g_W1 -= learning_rate * g_grad_W1
        g_b1 -= learning_rate * g_grad_b1
        g_W2 -= learning_rate * g_grad_W2
        g_b2 -= learning_rate * g_grad_b2
    
    # Return the generator forward function
    def gen_forward(z):
        hidden = relu(np.dot(z, g_W1) + g_b1)
        output = np.dot(hidden, g_W2) + g_b2
        return output, hidden, None  # Return 3 values to match expected interface
    
    return gen_forward