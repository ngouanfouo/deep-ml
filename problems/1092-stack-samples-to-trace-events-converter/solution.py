def stacks_to_trace(samples):
    if not samples:
        return []
    
    events = []
    prev_stack = None
    
    for timestamp, stack in samples:
        if prev_stack is None:
            # First sample: start everything
            for frame in stack:
                events.append((timestamp, 'start', frame))
        else:
            # Find longest common prefix
            common = 0
            min_len = min(len(prev_stack), len(stack))
            while common < min_len and prev_stack[common] == stack[common]:
                common += 1
            # End frames in prev beyond common, innermost first
            for frame in reversed(prev_stack[common:]):
                events.append((timestamp, 'end', frame))
            # Start frames in stack beyond common, outermost first
            for frame in stack[common:]:
                events.append((timestamp, 'start', frame))
        prev_stack = stack
    
    # End remaining frames at final timestamp, innermost first
    final_timestamp = samples[-1][0]
    for frame in reversed(prev_stack):
        events.append((final_timestamp, 'end', frame))
    
    return events