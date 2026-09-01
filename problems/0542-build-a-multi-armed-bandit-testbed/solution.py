import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    """
    np.random.seed(seed)
    
    # Generate true values
    true_values = np.random.randn(k)
    
    # Generate rewards for all arms at once (k x num_pulls matrix)
    # Each column has rewards for one arm
    rewards = np.random.randn(k, num_pulls) + true_values[:, np.newaxis]
    
    # Compute sample means across pulls (axis=1)
    sample_means = np.mean(rewards, axis=1)
    
    # Find optimal arm
    optimal_arm = int(np.argmax(true_values))
    
    # Round and return
    return np.round(true_values, 4).tolist(), np.round(sample_means, 4).tolist(), optimal_arm