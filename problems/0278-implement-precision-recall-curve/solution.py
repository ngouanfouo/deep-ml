import torch

def precision_recall_curve(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_scores: Tensor of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a torch.Tensor
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float64)
    if not isinstance(y_scores, torch.Tensor):
        y_scores = torch.tensor(y_scores, dtype=torch.float64)

    # Preserve input data type (e.g. float64) to prevent precision loss
    dtype = y_scores.dtype

    if len(y_true) != len(y_scores):
        raise ValueError(f"y_true and y_scores must have same length. Got {len(y_true)} and {len(y_scores)}")

    # Extract unique thresholds sorted in descending order
    thresholds = torch.unique(y_scores).sort(descending=True)[0]
    n_thresholds = len(thresholds)

    precisions = torch.zeros(n_thresholds, dtype=dtype)
    recalls = torch.zeros(n_thresholds, dtype=dtype)

    n_pos = (y_true == 1).sum().item()

    for i, t in enumerate(thresholds):
        preds = y_scores >= t
        tp = ((preds == 1) & (y_true == 1)).sum().item()
        fp = ((preds == 1) & (y_true == 0)).sum().item()

        # Handle precision edge case (no predicted positives)
        if tp + fp == 0:
            precisions[i] = 1.0
        else:
            precisions[i] = tp / (tp + fp)

        # Handle recall edge case (no actual positives in data)
        if n_pos == 0:
            recalls[i] = 0.0
        else:
            recalls[i] = tp / n_pos

    return precisions, recalls, thresholds