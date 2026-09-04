import torch

def calculate_inference_stats(latencies_ms: torch.Tensor) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: torch.Tensor of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if latencies_ms.numel() == 0:
        return {}

    # Convert to float for safe operations
    lat = latencies_ms.float()
    avg = lat.mean().item()
    throughput = 1000.0 / avg if avg > 0 else 0.0

    # Compute percentiles using linear interpolation (same as numpy default)
    p50 = torch.quantile(lat, 0.5, interpolation='linear').item()
    p95 = torch.quantile(lat, 0.95, interpolation='linear').item()
    p99 = torch.quantile(lat, 0.99, interpolation='linear').item()

    # Round to 2 decimal places
    return {
        'throughput_per_sec': round(throughput, 2),
        'avg_latency_ms': round(avg, 2),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2)
    }