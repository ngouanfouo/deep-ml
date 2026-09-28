def evaluate_preemption(
    num_tokens_list: list,
    kv_cache_bytes_per_token: int,
    swap_bandwidth_gbps: float,
    base_flops_per_token: int,
    attn_flops_per_token_pair: int,
    compute_tflops: float
) -> dict:
    """
    Evaluate preemption strategies (swap vs recompute) for a set of requests.

    Swap cost: move the KV cache to CPU and back.
        bytes_moved = 2 * num_tokens * kv_cache_bytes_per_token
        time_ms     = bytes_moved / (bandwidth_GBps * 1e9) * 1000

    Recompute cost: rebuild the KV cache from scratch.
        flops = num_tokens * base_flops_per_token
              + num_tokens^2 * attn_flops_per_token_pair
        time_ms = flops / (compute_TFLOPs * 1e12) * 1000
    """
    swap_costs_ms = []
    recompute_costs_ms = []
    strategies = []

    for n in num_tokens_list:
        # --- Swap cost ---
        bytes_moved = 2 * n * kv_cache_bytes_per_token
        swap_ms = bytes_moved / (swap_bandwidth_gbps * 1e9) * 1000.0

        # --- Recompute cost ---
        flops = n * base_flops_per_token + (n ** 2) * attn_flops_per_token_pair
        recompute_ms = flops / (compute_tflops * 1e12) * 1000.0

        swap_costs_ms.append(round(swap_ms, 4))
        recompute_costs_ms.append(round(recompute_ms, 4))

        # Tie -> prefer swap
        if swap_ms <= recompute_ms:
            strategies.append('swap')
        else:
            strategies.append('recompute')

    # Total cost uses the chosen (already rounded) values, matching the example
    total_cost_ms = round(
        sum(
            swap_costs_ms[i] if strategies[i] == 'swap' else recompute_costs_ms[i]
            for i in range(len(num_tokens_list))
        ),
        4
    )

    return {
        'strategies': strategies,
        'swap_costs_ms': swap_costs_ms,
        'recompute_costs_ms': recompute_costs_ms,
        'total_cost_ms': total_cost_ms,
    }