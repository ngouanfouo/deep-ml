def wave_pipeline_makespan(transfer_times, compute_times, send_times):
    """
    Compute the makespan of the three-stage wave pipeline.

    Args:
        transfer_times: list of per-wave token transfer durations
        compute_times: list of per-wave expert compute durations
        send_times: list of per-wave result send durations

    Returns:
        Total makespan (number).
    """
    n = len(transfer_times)
    
    # Initialize finish times for each stage
    transfer_finish = 0
    compute_finish = 0
    send_finish = 0
    
    for i in range(n):
        # Transfer stage must wait for:
        # 1. Previous wave's transfer to finish (transfer_finish)
        # 2. For wave 0, this is 0
        transfer_start = transfer_finish
        transfer_finish = transfer_start + transfer_times[i]
        
        # Compute stage must wait for:
        # 1. This wave's transfer to finish (transfer_finish)
        # 2. Previous wave's compute to finish (compute_finish)
        compute_start = max(transfer_finish, compute_finish)
        compute_finish = compute_start + compute_times[i]
        
        # Send stage must wait for:
        # 1. This wave's compute to finish (compute_finish)
        # 2. Previous wave's send to finish (send_finish)
        send_start = max(compute_finish, send_finish)
        send_finish = send_start + send_times[i]
    
    return send_finish