import torch

def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, normalize=None, round_decimals=4) -> torch.Tensor:
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Tensor or list of true labels in [0, K-1]
        y_pred: Tensor or list of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        torch.Tensor confusion matrix
    """
    # 1. Standardize inputs to flat 1D PyTorch tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.long)
    else:
        y_true = y_true.long().flatten()
        
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.long)
    else:
        y_pred = y_pred.long().flatten()

    # 2. Vectorized 2D index mapping using bincount
    # Map (row, col) to a flat index: row * K + col
    flat_indices = y_true * num_classes + y_pred
    
    # Compute counts across the entire possible range up to K^2
    counts = torch.bincount(flat_indices, minlength=num_classes * num_classes)
    
    # Reshape back to standard (num_classes, num_classes) structure
    cm = counts.reshape(num_classes, num_classes).float()

    # 3. Apply structural normalizations safely
    if normalize is not None:
        if normalize == 'true':
            # Row-wise normalization (divide by true class distribution counts)
            row_sums = cm.sum(dim=1, keepdim=True)
            cm = torch.where(row_sums > 0, cm / row_sums, torch.zeros_like(cm))
            
        elif normalize == 'pred':
            # Column-wise normalization (divide by predicted class distribution counts)
            col_sums = cm.sum(dim=0, keepdim=True)
            cm = torch.where(col_sums > 0, cm / col_sums, torch.zeros_like(cm))
            
        elif normalize == 'all':
            # Global normalization (divide by total dataset samples size)
            total_sum = cm.sum()
            if total_sum > 0:
                cm = cm / total_sum
                
        else:
            raise ValueError(f"Unknown normalization mode: {normalize}")
            
        # Round the resulting floating-point proportions
        cm = torch.round(cm, decimals=round_decimals)
        
    return cm