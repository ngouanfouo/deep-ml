import numpy as np

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        List of learning rates for each step from 0 to T-1
    """
    learning_rates = []
    
    for t in range(T):
        if t < W:
            # Linear Warmup phase: slope is lr_max / W
            lr = (t / W) * lr_max
        else:
            # Cosine Decay phase
            # Calculate how far we are into the decay phase
            decay_steps = T - W
            current_decay_step = t - W
            
            # Protect against edge case where T == W (no decay phase left)
            if decay_steps <= 0:
                lr = lr_max
            else:
                # Cosine annealing formula
                cosine_factor = 0.5 * (1 + np.cos(np.pi * current_decay_step / decay_steps))
                lr = lr_min + (lr_max - lr_min) * cosine_factor
                
        learning_rates.append(float(lr))
        
    return learning_rates