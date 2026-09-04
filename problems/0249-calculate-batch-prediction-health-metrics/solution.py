import torch

def calculate_batch_health(statuses: torch.Tensor, confidences: torch.Tensor, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        statuses: 1D tensor with 1 for 'success' and 0 for 'error'
        confidences: 1D tensor of confidence values (meaningful only for successful predictions)
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if statuses.numel() == 0:
        return {}

    total = statuses.size(0)
    success_mask = statuses == 1
    num_success = success_mask.sum().item()

    success_rate = (num_success / total) * 100.0

    if num_success == 0:
        avg_conf = 0.0
        low_rate = 0.0
    else:
        good_conf = confidences[success_mask]
        avg_conf = good_conf.mean().item() * 100.0
        low_conf_count = (good_conf < confidence_threshold).sum().item()
        low_rate = (low_conf_count / num_success) * 100.0

    return {
        'success_rate': round(success_rate, 2),
        'avg_confidence': round(avg_conf, 2),
        'low_confidence_rate': round(low_rate, 2)
    }