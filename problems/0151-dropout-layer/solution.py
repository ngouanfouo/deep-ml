import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        if not (0 <= p < 1):
            raise ValueError("p must be between 0 and 1 (exclusive of 1)")
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if training and self.p > 0:
            # Generate random mask: 1 for keep, 0 for drop
            keep_prob = 1 - self.p
            self.mask = np.random.binomial(1, keep_prob, size=x.shape)
            
            # Scale the remaining values by 1/keep_prob to maintain expected values
            return x * self.mask / keep_prob
        else:
            # Inference mode: pass input unchanged
            # Keep the mask from the last training forward pass
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent training forward pass.
        """
        if self.mask is None or self.p == 0:
            # If no mask was stored (no training forward pass yet or p=0), return gradient unchanged
            return grad
        
        keep_prob = 1 - self.p
        # Apply the same mask and scaling to the gradient
        return grad * self.mask / keep_prob