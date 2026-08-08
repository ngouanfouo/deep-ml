import torch

def calculate_auc(y_true, y_scores) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) using PyTorch.
    
    Args:
        y_true: List, array, or torch.Tensor of binary ground truth labels (0 or 1)
        y_scores: List, array, or torch.Tensor of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_scores, torch.Tensor):
        y_scores = torch.tensor(y_scores, dtype=torch.float32)
    
    # Ensure correct dtypes
    y_true = y_true.float()
    y_scores = y_scores.float()
    
    # Input validation
    if len(y_true) != len(y_scores):
        raise ValueError(f"y_true and y_scores must have same length. Got {len(y_true)} and {len(y_scores)}")
    
    # Handle edge case: all labels are the same
    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Sort by scores in descending order
    sorted_indices = torch.argsort(y_scores, descending=True)
    sorted_labels = y_true[sorted_indices]
    
    # Compute TPR and FPR
    # Initialize cumulative counts
    cum_tp = 0
    cum_fp = 0
    
    # Store points for ROC curve
    fpr_points = [0.0]
    tpr_points = [0.0]
    
    # Process each unique score threshold
    # We need to handle ties correctly: process all samples with the same score together
    unique_scores = torch.unique(y_scores)
    sorted_scores = torch.sort(unique_scores, descending=True)[0]
    
    for score in sorted_scores:
        # Get all samples with this score
        mask = y_scores == score
        # Count true positives and false positives among samples with this score
        tp_at_score = ((mask == 1) & (y_true == 1)).sum().item()
        fp_at_score = ((mask == 1) & (y_true == 0)).sum().item()
        
        # Update cumulative counts
        cum_tp += tp_at_score
        cum_fp += fp_at_score
        
        # Compute TPR and FPR at this threshold
        tpr = cum_tp / n_pos
        fpr = cum_fp / n_neg
        
        # Add point to ROC curve
        tpr_points.append(tpr)
        fpr_points.append(fpr)
    
    # Ensure we have (1,1) point at the end (when all samples are classified as positive)
    # Note: This should already be achieved at the lowest score threshold,
    # but we add it explicitly to be safe
    if fpr_points[-1] != 1.0 or tpr_points[-1] != 1.0:
        fpr_points.append(1.0)
        tpr_points.append(1.0)
    
    # Convert to tensors
    fpr_points = torch.tensor(fpr_points)
    tpr_points = torch.tensor(tpr_points)
    
    # Calculate AUC using trapezoidal integration
    # AUC = sum_{i=1}^{n} (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    # This is the trapezoidal rule for integration
    fpr_diff = fpr_points[1:] - fpr_points[:-1]
    tpr_sum = tpr_points[1:] + tpr_points[:-1]
    auc = torch.sum(fpr_diff * tpr_sum / 2)
    
    return auc.item()