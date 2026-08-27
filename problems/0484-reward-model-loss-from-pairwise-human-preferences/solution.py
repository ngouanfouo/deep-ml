import numpy as np

def reward_model_loss(chosen_rewards: list, rejected_rewards: list, margin: float = 0.0) -> dict:
    """
    Compute reward model loss from pairwise human preferences.
    
    Args:
        chosen_rewards: Reward scores for preferred responses
        rejected_rewards: Reward scores for non-preferred responses
        margin: Minimum desired gap between chosen and rejected scores
    
    Returns:
        Dictionary with 'loss' (float) and 'accuracy' (float)
    """
    # Convert to numpy arrays
    chosen = np.array(chosen_rewards)
    rejected = np.array(rejected_rewards)
    
    # Compute differences
    diff = chosen - rejected - margin
    
    # Numerically stable log_sigmoid: -log(1 + exp(-x))
    # Use np.where to handle positive and negative values differently
    # For x >= 0: -log(1 + exp(-x))
    # For x < 0: x - log(1 + exp(x))
    log_sigmoid = np.where(
        diff >= 0,
        -np.log1p(np.exp(-diff)),
        diff - np.log1p(np.exp(diff))
    )
    
    # Compute loss (negative average of log_sigmoid)
    loss = -np.mean(log_sigmoid)
    
    # Compute accuracy
    accuracy = np.mean(chosen > rejected)
    
    return {
        'loss': round(float(loss), 4),
        'accuracy': round(float(accuracy), 4)
    }