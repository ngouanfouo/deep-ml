def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.
    """
    ridge_point = peak_flops / peak_bandwidth

    def as_number(x):
        """Return int if x is a whole number, else float."""
        v = round(x, 4)
        if v == int(v):
            return int(v)
        return v

    def analyze(tokens):
        total_flops = 2.0 * num_params * tokens
        memory_bytes = float(num_params * bytes_per_param)
        arithmetic_intensity = total_flops / memory_bytes

        if arithmetic_intensity >= ridge_point:
            bottleneck = 'compute-bound'
            achieved_flops = peak_flops
        else:
            bottleneck = 'memory-bound'
            achieved_flops = arithmetic_intensity * peak_bandwidth

        utilization_percent = (achieved_flops / peak_flops) * 100.0

        return {
            'total_flops': as_number(total_flops),
            'memory_bytes': as_number(memory_bytes),
            'arithmetic_intensity': round(arithmetic_intensity, 4),
            'bottleneck': bottleneck,
            'achieved_flops': round(achieved_flops, 4),
            'utilization_percent': round(utilization_percent, 4)
        }

    return {
        'ridge_point': round(ridge_point, 4),
        'prefill': analyze(sequence_length),
        'decode': analyze(batch_size)
    }