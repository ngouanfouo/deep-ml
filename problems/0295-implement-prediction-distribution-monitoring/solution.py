import torch
import numpy as np
from scipy.stats import wasserstein_distance

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    This implementation includes additional metrics for more comprehensive drift detection.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    # Convert to tensors
    ref = torch.tensor(reference_preds, dtype=torch.float32)
    curr = torch.tensor(current_preds, dtype=torch.float32)
    
    # Validate inputs
    if len(ref) == 0 or len(curr) == 0:
        raise ValueError("Prediction lists cannot be empty")
    
    # Clip values
    ref = torch.clamp(ref, 0.0, 1.0)
    curr = torch.clamp(curr, 0.0, 1.0)
    
    # 1. Mean Shift
    mean_ref = torch.mean(ref)
    mean_curr = torch.mean(curr)
    mean_shift = (mean_curr - mean_ref).item()
    
    # 2. Standard Deviation Ratio
    std_ref = torch.std(ref, unbiased=True)  # Sample standard deviation
    std_curr = torch.std(curr, unbiased=True)
    
    if std_ref == 0:
        std_ratio = 1.0 if std_curr == 0 else float('inf')
    else:
        std_ratio = (std_curr / std_ref).item()
    
    # 3. Jensen-Shannon Divergence using histogram approach
    def compute_histogram(data, bins):
        """Compute histogram with fixed bin edges."""
        hist = torch.histc(data, bins=bins, min=0.0, max=1.0)
        return hist
    
    hist_ref = compute_histogram(ref, n_bins)
    hist_curr = compute_histogram(curr, n_bins)
    
    # Laplace smoothing
    hist_ref_smooth = hist_ref + 1.0
    hist_curr_smooth = hist_curr + 1.0
    
    # Normalize to probability distributions
    p = hist_ref_smooth / torch.sum(hist_ref_smooth)
    q = hist_curr_smooth / torch.sum(hist_curr_smooth)
    
    # Compute JS divergence
    def kl_divergence(p, q):
        """Compute KL divergence with epsilon for numerical stability."""
        eps = 1e-10
        return torch.sum(p * torch.log((p + eps) / (q + eps)))
    
    m = (p + q) / 2
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)
    js_divergence = 0.5 * (kl_p_m + kl_q_m).item()
    
    # 4. Drift Detection
    drift_threshold = 0.1
    drift_detected = js_divergence > drift_threshold
    
    return {
        'mean_shift': round(mean_shift, 6),
        'std_ratio': round(std_ratio, 6) if std_ratio != float('inf') else float('inf'),
        'js_divergence': round(js_divergence, 6),
        'drift_detected': drift_detected
    }