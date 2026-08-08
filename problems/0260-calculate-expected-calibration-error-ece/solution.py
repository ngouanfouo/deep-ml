import torch

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or tensor of true binary labels (0 or 1)
        y_prob: List or tensor of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    # Convert to torch tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_prob, torch.Tensor):
        y_prob = torch.tensor(y_prob, dtype=torch.float32)
    
    # Ensure y_true is float for calculations
    y_true = y_true.float()
    y_prob = y_prob.float()
    
    # Create bin boundaries
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    
    # Assign each sample to a bin
    # For probabilities equal to 0, we want them in the first bin
    # For other probabilities, use right=True to get (lower, upper] bins
    bin_indices = torch.searchsorted(bin_edges, y_prob, right=True) - 1
    
    # Clamp indices to ensure valid range (handles prob=0 and prob=1 cases)
    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
    
    # Calculate bin statistics
    # For each bin, we need sum of confidences, sum of accuracies, and count
    bin_counts = torch.zeros(n_bins, dtype=torch.float32)
    bin_confidences = torch.zeros(n_bins, dtype=torch.float32)
    bin_accuracies = torch.zeros(n_bins, dtype=torch.float32)
    
    # Use scatter_add to accumulate statistics per bin
    bin_counts.scatter_add_(0, bin_indices, torch.ones_like(y_prob, dtype=torch.float32))
    bin_confidences.scatter_add_(0, bin_indices, y_prob)
    bin_accuracies.scatter_add_(0, bin_indices, y_true)
    
    # Calculate ECE
    n_samples = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_confidence = bin_confidences[i] / bin_counts[i]
            avg_accuracy = bin_accuracies[i] / bin_counts[i]
            weight = bin_counts[i] / n_samples
            ece += weight * torch.abs(avg_accuracy - avg_confidence)
    
    return round(ece.item(), 3)