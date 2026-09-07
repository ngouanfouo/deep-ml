def allocator_ops(num_blocks, operations):
    """
    Simulates a fixed-size block memory allocator.

    Args:
        num_blocks (int): Total number of fixed-size blocks.
        operations (list of tuples): Sequence of 'alloc' and 'free' operations.

    Returns:
        list: Results for each operation (int for alloc, bool for free).
    """
    allocated = [False] * num_blocks
    free_stack = []
    next_untouched = 0
    results = []

    for op in operations:
        cmd = op[0]
        if cmd == "alloc":
            if free_stack:
                idx = free_stack.pop()
            elif next_untouched < num_blocks:
                idx = next_untouched
                next_untouched += 1
            else:
                results.append(-1)
                continue
            
            allocated[idx] = True
            results.append(idx)

        elif cmd == "free":
            _, idx = op
            if 0 <= idx < num_blocks and allocated[idx]:
                allocated[idx] = False
                free_stack.append(idx)
                results.append(True)
            else:
                results.append(False)

    return results