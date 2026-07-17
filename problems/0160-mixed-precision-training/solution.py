import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        # Initialize loss scaling factor
        self.loss_scale = loss_scale
    
    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, return scaled loss as float32
        # Convert to float16 for computation
        weights_f16 = weights.astype(np.float16)
        inputs_f16 = inputs.astype(np.float16)
        targets_f16 = targets.astype(np.float16)
        
        # Forward pass: compute predictions (linear regression)
        # predictions = inputs @ weights
        predictions = np.dot(inputs_f16, weights_f16)
        
        # Compute MSE loss: mean((predictions - targets)^2)
        # Use float16 for computation
        diff = predictions - targets_f16
        mse_loss = np.mean(diff ** 2)
        
        # Scale the loss and convert to float32
        scaled_loss = float(mse_loss * self.loss_scale)
        
        return scaled_loss
    
    def backward(self, gradients):
        # Unscale gradients and check for overflow, return as float32
        
        # Convert gradients to float32 and unscale
        gradients_f32 = gradients.astype(np.float32)
        unscaled_grads = gradients_f32 / self.loss_scale
        
        # Check for overflow (NaN or Inf)
        if np.any(np.isnan(unscaled_grads)) or np.any(np.isinf(unscaled_grads)):
            # If overflow detected, return zero gradients
            return np.zeros_like(unscaled_grads, dtype=np.float32)
        
        # Return unscaled gradients as float32
        return unscaled_grads