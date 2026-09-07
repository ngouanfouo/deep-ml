from collections import OrderedDict
import functools


def _make_hashable(val):
    """Recursively converts unhashable containers (lists, dicts, sets) into hashable equivalents."""
    if isinstance(val, (list, tuple)):
        return tuple(_make_hashable(x) for x in val)
    elif isinstance(val, dict):
        try:
            sorted_items = sorted(val.items())
        except TypeError:
            sorted_items = sorted(val.items(), key=lambda item: (str(type(item[0])), repr(item[0])))
        return tuple((k, _make_hashable(v)) for k, v in sorted_items)
    elif isinstance(val, (set, frozenset)):
        try:
            sorted_elems = sorted(_make_hashable(x) for x in val)
        except TypeError:
            sorted_elems = sorted([_make_hashable(x) for x in val], key=lambda x: (str(type(x)), repr(x)))
        return frozenset(sorted_elems)
    else:
        return val


def lru_memoize(capacity):
    """Returns a decorator that wraps a function with an O(1) LRU memoization cache."""
    def decorator(func):
        cache = OrderedDict()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Normalize arguments into a canonical hashable key
            hashable_args = _make_hashable(args)
            hashable_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
            key = (hashable_args, hashable_kwargs)
            
            if key in cache:
                # Cache hit: mark as recently used
                cache.move_to_end(key)
                return cache[key]
            
            # Cache miss: compute result
            result = func(*args, **kwargs)
            cache[key] = result
            
            # Evict least recently used item if capacity is exceeded
            if len(cache) > capacity:
                cache.popitem(last=False)
                
            return result
            
        wrapper.cache = cache
        return wrapper
    return decorator