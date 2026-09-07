from collections import OrderedDict


def lru_cache_operations(capacity, ops):
    """
    Implements a fixed-capacity LRU cache supporting put, get, and crash operations.
    
    Args:
        capacity (int): Maximum number of items the cache can hold.
        ops (list): List of operations (['put', key, val], ['get', key], ['crash']).
        
    Returns:
        list: Results of every 'get' operation in order.
    """
    cache = OrderedDict()
    outputs = []
    
    for op in ops:
        action = op[0]
        if action == "put":
            key, val = op[1], op[2]
            if key in cache:
                cache.move_to_end(key)
            cache[key] = val
            if len(cache) > capacity:
                # Evict the least recently used item (the first item)
                cache.popitem(last=False)
        elif action == "get":
            key = op[1]
            if key in cache:
                cache.move_to_end(key)
                outputs.append(cache[key])
            else:
                outputs.append(-1)
        elif action == "crash":
            # Per instructions, the crash preserves all data and recency ordering fully.
            pass
            
    return outputs