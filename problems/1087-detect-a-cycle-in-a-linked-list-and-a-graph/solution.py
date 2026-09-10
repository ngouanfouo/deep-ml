def detect_cycles(ll_next, graph):
    # --- Part 1: Linked list cycle entry (Floyd's) ---
    entry = -1
    if ll_next:
        # Phase 1: detect cycle
        slow = 0
        fast = 0
        met = False
        while fast != -1:
            slow = ll_next[slow]
            fast = ll_next[fast]
            if fast != -1:
                fast = ll_next[fast]
            else:
                break
            if slow == fast:
                met = True
                break
        
        if met:
            # Phase 2: find entry
            slow = 0
            while slow != fast:
                slow = ll_next[slow]
                fast = ll_next[fast]
            entry = slow
    
    # --- Part 2: Directed graph cycle detection ---
    n = len(graph)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    has_cycle = False
    
    def dfs(u):
        nonlocal has_cycle
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                has_cycle = True
                return
            if color[v] == WHITE:
                dfs(v)
                if has_cycle:
                    return
        color[u] = BLACK
    
    for i in range(n):
        if color[i] == WHITE:
            dfs(i)
            if has_cycle:
                break
    
    return (entry, has_cycle)