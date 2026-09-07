import bisect

def snapshot_array(length, operations):
    """
    Simulates a versioned array supporting set, snap, and get operations.
    
    Args:
        length (int): Size of the array.
        operations (list): List of operations (['set', index, val], ['snap'], ['get', index, snap_id]).
        
    Returns:
        list: All outputs produced by snap and get operations in order.
    """
    snap_ids = [[0] for _ in range(length)]
    values = [[0] for _ in range(length)]
    current_snap_id = 0
    output = []
    
    for op in operations:
        action = op[0]
        if action == "set":
            index, val = op[1], op[2]
            # If a value was already set during the current snapshot period, update it in place
            if snap_ids[index][-1] == current_snap_id:
                values[index][-1] = val
            else:
                snap_ids[index].append(current_snap_id)
                values[index].append(val)
        elif action == "snap":
            output.append(current_snap_id)
            current_snap_id += 1
        elif action == "get":
            index, snap_id = op[1], op[2]
            # Use binary search to find the latest value at or before the given snap_id
            idx = bisect.bisect_right(snap_ids[index], snap_id) - 1
            output.append(values[index][idx])
            
    return output