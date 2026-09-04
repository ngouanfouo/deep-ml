import torch

def calculate_sla_metrics(latencies: torch.Tensor, statuses: torch.Tensor, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        latencies: 1D tensor of latency values in ms
        statuses: 1D tensor of status codes (0=success, 1=error, 2=timeout)
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if latencies.numel() == 0:
        return {}

    total = latencies.size(0)

    # statuses: 0 = success, 1 = error, 2 = timeout
    success_mask = statuses == 0
    error_mask = statuses != 0

    success_count = success_mask.sum().item()
    error_count = error_mask.sum().item()

    # Successful requests that meet latency SLA
    if success_count > 0:
        ok_success = (latencies <= latency_sla_ms) & success_mask
        ok_count = ok_success.sum().item()
        latency_sla_compliance = (ok_count / success_count) * 100.0
    else:
        ok_count = 0
        latency_sla_compliance = 0.0

    error_rate = (error_count / total) * 100.0
    overall_sla_compliance = (ok_count / total) * 100.0

    return {
        'latency_sla_compliance': round(latency_sla_compliance, 2),
        'error_rate': round(error_rate, 2),
        'overall_sla_compliance': round(overall_sla_compliance, 2)
    }