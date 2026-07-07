import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # 1. Calculate the clean/base logits
    clean_logits = X @ W_g
    
    # 2. Calculate the noise scaling factor using Softplus: log(1 + exp(x))
    noise_logits = X @ W_noise
    noise_scale = np.log1p(np.exp(noise_logits))
    
    # 3. Add the pre-sampled noise scaled by the noise factor
    noisy_logits = clean_logits + (N * noise_scale)
    
    # 4. Enforce Sparsity: Find the k-th largest value for each row
    if k < noisy_logits.shape[1]:
        # Sort along the expert dimension to find the threshold per row
        sorted_logits = np.sort(noisy_logits, axis=-1)
        # The k-th largest element threshold
        thresholds = sorted_logits[:, -k][:, np.newaxis]
        
        # Mask out anything below the top-k threshold by setting to -inf
        mask = noisy_logits < thresholds
        noisy_logits = np.where(mask, -np.inf, noisy_logits)
        
    # 5. Compute Stable Softmax
    # Subtract max for numerical stability against overflow
    max_logits = np.max(noisy_logits, axis=-1, keepdims=True)
    # Handle possible -inf components safely during max subtraction
    max_logits = np.where(max_logits == -np.inf, 0, max_logits)
    
    exp_logits = np.exp(noisy_logits - max_logits)
    gating_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return gating_probs