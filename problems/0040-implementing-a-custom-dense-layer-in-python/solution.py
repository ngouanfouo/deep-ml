import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.W_opt = None
        self.w0_opt = None

    def initialize(self, optimizer):
        """
        Initialize the weights using a uniform distribution bounded by 1 / sqrt(input_features),
        and set the biases to zero.
        """
        input_features = self.input_shape[0]
        limit = 1 / math.sqrt(input_features)
        
        # Initialize weights and biases
        self.W = np.random.uniform(-limit, limit, (input_features, self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        
        # Allocate deep copies of the optimizer for weights and biases separately
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        """Return the total count of trainable parameters (weights + biases)."""
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        """
        Compute the linear activation mapping out = X * W + w0.
        Cache the input state for usage during the backward gradient pass.
        """
        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        """
        Calculate loss derivatives with respect to the inputs, weights, and biases.
        Update parameters using the optimizers if the layer is trainable.
        """
        # Save the current state of W to compute the input gradient safely
        W_current = self.W
        
        if self.trainable:
            # Calculate gradients: dW = X^T * accum_grad; dw0 = sum(accum_grad, axis=0)
            dW = np.dot(self.layer_input.T, accum_grad)
            dw0 = np.sum(accum_grad, axis=0, keepdims=True)
            
            # Step parameters utilizing their independent optimizer instances
            self.W = self.W_opt.update(self.W, dW)
            self.w0 = self.w0_opt.update(self.w0, dw0)
            
        # Return gradient with respect to input: accum_grad * W^T
        return np.dot(accum_grad, W_current.T)

    def output_shape(self):
        """Return the shape dimensions of a single prediction vector instance."""
        return (self.n_units,)