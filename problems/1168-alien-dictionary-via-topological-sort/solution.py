import heapq

def alien_order(words):
    """
    Returns a string containing all distinct characters in words,
    ordered consistently with the alien language's sorting rules.
    """
    # 1. Collect all unique characters
    unique_chars = set()
    for word in words:
        for char in word:
            unique_chars.add(char)
            
    # Initialize graph and in-degree count for all unique characters
    graph = {char: set() for char in unique_chars}
    in_degree = {char: 0 for char in unique_chars}
    
    # 2. Build the dependency graph from adjacent word pairs
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        
        # Check invalid condition: a longer word comes before its own strict prefix
        if len(w1) > len(w2) and w1[:len(w2)] == w2:
            return ""
            
        min_len = min(len(w1), len(w2))
        for j in range(min_len):
            if w1[j] != w2[j]:
                c1, c2 = w1[j], w2[j]
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break  # Only the first differing character matters for ordering
                
    # 3. Kahn's Algorithm with a min-heap for alphabetical tie-breaking
    heap = [char for char in unique_chars if in_degree[char] == 0]
    heapq.heapify(heap)
    
    res = []
    while heap:
        curr = heapq.heappop(heap)
        res.append(curr)
        
        for neighbor in graph[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)
                
    # If not all characters are visited, there is a cycle (invalid ordering)
    if len(res) != len(unique_chars):
        return ""
        
    return "".join(res)