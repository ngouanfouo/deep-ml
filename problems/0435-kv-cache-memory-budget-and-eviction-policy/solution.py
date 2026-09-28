def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int,
                     dtype_bytes: int, memory_budget_bytes: int,
                     token_ids: list, token_scores: list,
                     eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with a memory budget and an eviction policy.
    """
    # --- Memory math ---------------------------------------------------
    # Keys + values, each of shape (num_layers, num_heads, head_dim),
    # stored at `dtype_bytes` per element.
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    max_tokens = memory_budget_bytes // bytes_per_token if bytes_per_token > 0 else 0

    # --- Simulation ----------------------------------------------------
    # Each entry: {"id": int, "score": float, "protected": bool}
    cache = []
    evicted_tokens = []

    for position, (tid, score) in enumerate(zip(token_ids, token_scores)):
        is_protected = position < num_protected

        # Cache full? Try to evict a non-protected entry first.
        if len(cache) >= max_tokens:
            candidates = [(i, e) for i, e in enumerate(cache) if not e["protected"]]

            if not candidates:
                # Everything is protected — the new token cannot be added.
                continue

            if eviction_policy == "fifo":
                # Oldest non-protected entry = earliest in insertion order.
                evict_idx = candidates[0][0]
            elif eviction_policy == "score":
                # Lowest score; ties broken by oldest (smallest index).
                evict_idx = min(
                    candidates,
                    key=lambda pair: (pair[1]["score"], pair[0]),
                )[0]
            else:
                raise ValueError(
                    f"eviction_policy must be 'fifo' or 'score', got {eviction_policy!r}"
                )

            evicted_tokens.append(cache[evict_idx]["id"])
            cache.pop(evict_idx)

        # Insert the new token.
        cache.append({"id": tid, "score": score, "protected": is_protected})

    return {
        "bytes_per_token": int(bytes_per_token),
        "max_tokens": int(max_tokens),
        "final_cache": [e["id"] for e in cache],
        "evicted_tokens": evicted_tokens,
        "num_evictions": len(evicted_tokens),
    }