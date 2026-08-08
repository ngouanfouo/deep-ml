def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    # Input validation
    if not requests:
        return []
    
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    
    if max_wait_time <= 0:
        raise ValueError("max_wait_time must be positive")
    
    # Sort requests by timestamp
    sorted_requests = sorted(requests, key=lambda x: x['timestamp'])
    
    batches = []
    current_batch_ids = []
    current_batch_features = []
    batch_start_time = sorted_requests[0]['timestamp']
    
    for request in sorted_requests:
        # Check if adding this request would violate constraints
        time_elapsed = request['timestamp'] - batch_start_time
        should_start_new_batch = (
            len(current_batch_ids) >= max_batch_size or
            time_elapsed > max_wait_time
        )
        
        if should_start_new_batch and current_batch_ids:
            # Finalize current batch
            # Process time is the timestamp of the last request in the batch
            # which is the previous request's timestamp
            process_time = round(current_batch_ids[-1]['timestamp'] if hasattr(current_batch_ids[-1], '__getitem__') 
                                 else sorted_requests[sorted_requests.index(request) - 1]['timestamp'], 4)
            
            # Actually, we should use the previous request's timestamp
            # Find the previous request
            prev_request = None
            for i, r in enumerate(sorted_requests):
                if r['id'] == request['id'] and i > 0:
                    prev_request = sorted_requests[i - 1]
                    break
            
            if prev_request is None and current_batch_ids:
                # If we can't find previous, use the last request in batch
                # But we don't have it directly, so use the start time + max_wait_time
                # Actually, process_time should be the timestamp of the last request
                # that triggered the batch finalization
                if len(current_batch_ids) >= max_batch_size:
                    # Batch filled by size, use the timestamp of the last request in batch
                    # This is the current request's timestamp if it's the one that filled it
                    process_time = round(request['timestamp'], 4)
                else:
                    # Batch filled by time, use the timestamp when max_wait_time was reached
                    process_time = round(batch_start_time + max_wait_time, 4)
            else:
                process_time = round(prev_request['timestamp'], 4)
            
            batches.append((
                current_batch_ids,
                current_batch_features,
                process_time
            ))
            
            # Start new batch with current request
            current_batch_ids = [request['id']]
            current_batch_features = [request['features']]
            batch_start_time = request['timestamp']
        else:
            # Add to current batch
            current_batch_ids.append(request['id'])
            current_batch_features.append(request['features'])
    
    # Handle the last batch
    if current_batch_ids:
        process_time = round(sorted_requests[-1]['timestamp'], 4)
        batches.append((
            current_batch_ids,
            current_batch_features,
            process_time
        ))
    
    return batches