def dedup_components(n, pairs):
    """
    Group image ids 0..n-1 into connected components from duplicate pairs
    using union-find, and return a dict {image_id: representative}, where
    each representative is the smallest id in its component.
    """
    if n == 0:
        return {}
    
    # parent[i] = i initially; we always attach the larger root under the smaller
    parent = list(range(n))
    
    def find(x):
        # Path compression (iterative, to avoid recursion depth issues)
        root = x
        while parent[root] != root:
            root = parent[root]
        # Compress the path
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Make the smaller root the parent of the larger root
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb
    
    # Process all pairs (self-loops and duplicate edges are harmless)
    for a, b in pairs:
        union(a, b)
    
    # Build result in increasing key order 0..n-1
    result = {}
    for i in range(n):
        result[i] = find(i)
    
    return result