import torch

def compute_arithmetic_intensity(
    flops: torch.Tensor,
    bytes_accessed: torch.Tensor,
    peak_performance: torch.Tensor,
    peak_bandwidth: torch.Tensor
) -> dict:
    """
    Analyze a computational kernel using the Roofline Model with PyTorch tensors.
    """
    # Convert to Python floats
    flops_val = flops.item()
    bytes_val = bytes_accessed.item()
    peak_perf = peak_performance.item()
    peak_bw = peak_bandwidth.item()

    # Arithmetic intensity
    arithmetic_intensity = flops_val / bytes_val

    # Ridge point
    ridge_point = peak_perf / peak_bw

    # Bottleneck classification
    if arithmetic_intensity < ridge_point:
        bottleneck = 'memory-bound'
        achieved_performance = arithmetic_intensity * peak_bw
    else:
        bottleneck = 'compute-bound'
        achieved_performance = peak_perf

    # Utilization
    utilization_percent = (achieved_performance / peak_perf) * 100.0

    return {
        'arithmetic_intensity': round(arithmetic_intensity, 4),
        'ridge_point': round(ridge_point, 4),
        'bottleneck': bottleneck,
        'achieved_performance': round(achieved_performance, 4),
        'utilization_percent': round(utilization_percent, 4)
    }