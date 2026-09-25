import math

def cow_memory_simulator(prompt_length: int, num_sequences: int, gen_lengths: list, block_size: int) -> dict:
    """
    Simulate copy-on-write memory management for parallel LLM sampling.
    """
    # 1. Split prompt into full blocks (always shared) + optional partial.
    full_prompt_blocks = prompt_length // block_size
    remainder = prompt_length % block_size
    shared_prompt_blocks = full_prompt_blocks

    if remainder > 0:
        free_slots = block_size - remainder

        # The partial block starts with refcount == num_sequences
        # (every sequence references the prompt).  Each writer either
        # CoWs (if refcount > 1) or takes ownership (if refcount == 1).
        # Result: cow_events = min(writers, num_sequences - 1).
        num_writers = sum(1 for g in gen_lengths if g >= 1)
        cow_events = min(num_writers, max(0, num_sequences - 1))

        # 1 original block + 1 new physical block per CoW event.
        partial_physical_blocks = 1 + cow_events

        # 3. Spill-over: tokens beyond the partial block's free slots.
        new_generation_blocks = 0
        for g in gen_lengths:
            if g == 0:
                continue
            leftover = g - min(g, free_slots)
            if leftover > 0:
                new_generation_blocks += math.ceil(leftover / block_size)
    else:
        # No partial block — every generated token needs a fresh block.
        partial_physical_blocks = 0
        cow_events = 0
        new_generation_blocks = 0
        for g in gen_lengths:
            if g > 0:
                new_generation_blocks += math.ceil(g / block_size)

    # 4. Totals.
    total_physical_blocks = (
        shared_prompt_blocks + partial_physical_blocks + new_generation_blocks
    )
    total_logical_blocks = sum(
        math.ceil((prompt_length + g) / block_size) for g in gen_lengths
    )

    if total_logical_blocks > 0:
        memory_saved_pct = round(
            (1.0 - total_physical_blocks / total_logical_blocks) * 100.0, 2
        )
    else:
        memory_saved_pct = 0.0

    return {
        'total_physical_blocks': total_physical_blocks,
        'total_logical_blocks': total_logical_blocks,
        'shared_prompt_blocks': shared_prompt_blocks,
        'cow_events': cow_events,
        'new_generation_blocks': new_generation_blocks,
        'memory_saved_pct': memory_saved_pct,
    }