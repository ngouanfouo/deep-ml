import torch

def compute_roc_curve(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    This implementation uses a more efficient approach by sorting predictions
    and computing TPR and FPR incrementally.
    
    Args:
        y_true: Binary ground truth labels tensor (0 or 1)
        y_scores: Predicted scores/probabilities tensor for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a torch.Tensor of floats
    """
    # Input validation
    if len(y_true) != len(y_scores):
        raise ValueError(f"y_true and y_scores must have same length. Got {len(y_true)} and {len(y_scores)}")
    
    # Convert to tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_scores, torch.Tensor):
        y_scores = torch.tensor(y_scores, dtype=torch.float32)
    
    y_true = y_true.float()
    y_scores = y_scores.float()
    
    # Get total positives and negatives
    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()
    
    # Handle edge cases
    if n_pos == 0:
        # No positive samples
        thresholds = torch.unique(y_scores).sort(descending=True)[0]
        thresholds = torch.cat([torch.tensor([float('inf')]), thresholds])
        return torch.ones_like(thresholds), torch.zeros_like(thresholds)
    
    if n_neg == 0:
        # No negative samples
        thresholds = torch.unique(y_scores).sort(descending=True)[0]
        thresholds = torch.cat([torch.tensor([float('inf')]), thresholds])
        return torch.zeros_like(thresholds), torch.ones_like(thresholds)
    
    # Sort scores and labels by score in descending order
    sorted_indices = torch.argsort(y_scores, descending=True)
    sorted_y_true = y_true[sorted_indices]
    
    # Get unique thresholds (scores)
    unique_scores = torch.unique(y_scores)
    thresholds = unique_scores.sort(descending=True)[0]
    thresholds = torch.cat([torch.tensor([float('inf')]), thresholds])
    
    # Initialize FPR and TPR
    fpr = torch.zeros(len(thresholds))
    tpr = torch.zeros(len(thresholds))
    
    # Track cumulative true positives and false positives
    cum_tp = 0
    cum_fp = 0
    
    # Process each unique threshold
    for i, threshold in enumerate(thresholds):
        if threshold == float('inf'):
            # No samples classified as positive
            fpr[i] = 0.0
            tpr[i] = 0.0
        else:
            # Find all samples with score >= threshold
            # Since scores are sorted, this is all samples up to the current index
            # But we need to handle duplicate thresholds correctly
            # We'll process each threshold by counting samples >= threshold
            mask = y_scores >= threshold
            tp = ((mask == 1) & (y_true == 1)).sum().item()
            fp = ((mask == 1) & (y_true == 0)).sum().item()
            
            fpr[i] = fp / n_neg
            tpr[i] = tp / n_pos
    
    return fpr, tpr