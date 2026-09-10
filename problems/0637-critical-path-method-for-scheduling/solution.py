import heapq

def critical_path_method(tasks, dependencies):
    # Build successor lists and in-degrees
    successors = {t: [] for t in tasks}
    in_degree = {t: 0 for t in tasks}
    for t, deps in dependencies.items():
        for d in deps:
            successors[d].append(t)
            in_degree[t] += 1
    
    # Topological sort with alphabetical tiebreak
    heap = [t for t in tasks if in_degree[t] == 0]
    heapq.heapify(heap)
    topo_order = []
    while heap:
        t = heapq.heappop(heap)
        topo_order.append(t)
        for s in sorted(successors[t]):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                heapq.heappush(heap, s)
    
    # Forward pass: ES, EF
    ES = {}
    EF = {}
    for t in topo_order:
        if dependencies[t]:
            ES[t] = max(EF[d] for d in dependencies[t])
        else:
            ES[t] = 0
        EF[t] = ES[t] + tasks[t]
    
    project_duration = max(EF.values()) if EF else 0
    
    # Backward pass: LF, LS
    LF = {}
    LS = {}
    for t in reversed(topo_order):
        if successors[t]:
            LF[t] = min(LS[s] for s in successors[t])
        else:
            LF[t] = project_duration
        LS[t] = LF[t] - tasks[t]
    
    # Slack
    slack = {t: LS[t] - ES[t] for t in tasks}
    
    # Critical path: zero-slack tasks in topological order
    critical_path = [t for t in topo_order if slack[t] == 0]
    
    return {
        'project_duration': project_duration,
        'critical_path': critical_path,
        'slack': slack,
    }