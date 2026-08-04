import torch

def analyze_canary_deployment(
    canary_latencies: torch.Tensor,
    canary_predictions: torch.Tensor,
    canary_ground_truths: torch.Tensor,
    baseline_latencies: torch.Tensor,
    baseline_predictions: torch.Tensor,
    baseline_ground_truths: torch.Tensor,
    accuracy_tolerance: float = 0.05,
    latency_tolerance: float = 0.10
) -> dict:
    """
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_latencies: Tensor of latencies from canary model (ms)
        canary_predictions: Tensor of predictions from canary model
        canary_ground_truths: Tensor of ground truth values for canary
        baseline_latencies: Tensor of latencies from baseline model (ms)
        baseline_predictions: Tensor of predictions from baseline model
        baseline_ground_truths: Tensor of ground truth values for baseline
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    # Check for empty inputs
    if (len(canary_latencies) == 0 or len(baseline_latencies) == 0 or
        len(canary_predictions) == 0 or len(baseline_predictions) == 0 or
        len(canary_ground_truths) == 0 or len(baseline_ground_truths) == 0):
        return {}
    
    # Ensure all tensors have the same length for each model
    # Canary
    canary_correct = (canary_predictions == canary_ground_truths).float()
    canary_accuracy = torch.mean(canary_correct).item()
    
    # Baseline
    baseline_correct = (baseline_predictions == baseline_ground_truths).float()
    baseline_accuracy = torch.mean(baseline_correct).item()
    
    # Calculate accuracy change percentage
    # Relative change: ((new - old) / old) * 100
    accuracy_change_pct = ((canary_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    
    # Calculate average latencies
    canary_avg_latency = torch.mean(canary_latencies.float()).item()
    baseline_avg_latency = torch.mean(baseline_latencies.float()).item()
    
    # Calculate latency change percentage
    # Relative change: ((new - old) / old) * 100
    latency_change_pct = ((canary_avg_latency - baseline_avg_latency) / baseline_avg_latency) * 100
    
    # Determine if promotion is recommended
    # Promote if accuracy didn't degrade beyond tolerance AND latency didn't increase beyond tolerance
    # Accuracy degradation: canary_accuracy >= baseline_accuracy * (1 - accuracy_tolerance)
    # Latency increase: canary_avg_latency <= baseline_avg_latency * (1 + latency_tolerance)
    
    accuracy_ok = canary_accuracy >= baseline_accuracy * (1 - accuracy_tolerance)
    latency_ok = canary_avg_latency <= baseline_avg_latency * (1 + latency_tolerance)
    promote_recommended = accuracy_ok and latency_ok
    
    # Round all values
    return {
        'canary_accuracy': round(canary_accuracy, 4),
        'baseline_accuracy': round(baseline_accuracy, 4),
        'accuracy_change_pct': round(accuracy_change_pct, 2),
        'canary_avg_latency': round(canary_avg_latency, 2),
        'baseline_avg_latency': round(baseline_avg_latency, 2),
        'latency_change_pct': round(latency_change_pct, 2),
        'promote_recommended': bool(promote_recommended)
    }