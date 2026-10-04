def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
    """
    Perform Roofline Model analysis for GPU operations.

    Args:
        peak_gflops: Peak compute throughput in GFLOPS
        peak_bandwidth_gbs: Peak memory bandwidth in GB/s
        operations: List of dicts with keys 'name', 'flops', 'bytes'

    Returns:
        Dict with 'ridge_point' and 'operations' list containing
        per-operation analysis results.
    """
    # Ridge point: hardware transition between memory-bound and compute-bound
    ridge_point = peak_gflops / peak_bandwidth_gbs

    results = []
    for op in operations:
        name = op['name']
        flops = op['flops']
        bytes_ = op['bytes']

        # Operational intensity (FLOP/byte)
        operational_intensity = flops / bytes_

        # Attainable performance: min of compute roof and memory roof
        attainable_gflops = min(
            peak_gflops,
            peak_bandwidth_gbs * operational_intensity
        )

        # Bottleneck classification
        if operational_intensity >= ridge_point:
            bottleneck = 'compute-bound'
        else:
            bottleneck = 'memory-bound'

        # Efficiency as percentage of peak compute
        efficiency = (attainable_gflops / peak_gflops) * 100.0

        results.append({
            'name': name,
            'operational_intensity': operational_intensity,
            'attainable_gflops': attainable_gflops,
            'bottleneck': bottleneck,
            'efficiency': efficiency
        })

    return {
        'ridge_point': ridge_point,
        'operations': results
    }