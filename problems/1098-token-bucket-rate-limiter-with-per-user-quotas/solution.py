def simulate(events, capacity, refill_rate, queue_capacity):
    """
    Simulates an admission layer with per-user token buckets and a shared FIFO queue.

    Args:
        events (list): List of (time, type, user) tuples where type is "produce" or "consume".
        capacity (float/int): Maximum token capacity per user bucket.
        refill_rate (float/int): Tokens added per unit of time.
        queue_capacity (int): Maximum items the shared queue can hold.

    Returns:
        list: Result strings corresponding to each event in order.
    """
    user_state = {}  # user -> [tokens, last_seen_time]
    queue = []
    results = []
    
    for time, type_, user in events:
        if type_ == "produce":
            if user not in user_state:
                # First time seen: starts FULL with capacity tokens, no refill
                tokens = float(capacity)
                last_seen = time
            else:
                tokens, last_seen = user_state[user]
                delta = time - last_seen
                tokens = min(float(capacity), tokens + delta * refill_rate)
                last_seen = time
            
            user_state[user] = [tokens, last_seen]
            
            if tokens < 1.0:
                results.append("limited")
            elif len(queue) >= queue_capacity:
                results.append("full")
            else:
                # Spend 1 token and add to the queue
                user_state[user][0] -= 1.0
                queue.append(1)
                results.append("ok")
                
        elif type_ == "consume":
            if not queue:
                results.append("empty")
            else:
                queue.pop(0)
                results.append("item")
                
    return results