def _topo_order(graph, outputs):
    visited = set()
    order = []

    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        if n in graph:
            op = graph[n][0]
            if op != "buf":
                for inp in graph[n][1:]:
                    dfs(inp)
        order.append(n)

    for o in outputs:
        dfs(o)
    return order


def _realized(graph, outputs):
    real = set(outputs)
    for n, node in graph.items():
        op = node[0]
        if op == "buf" or op == "reduce":
            real.add(n)
    return real


def _build_kernel(graph, R, real):
    fused = set()
    inputs = set()
    visited = set()
    stack = [R]
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n not in graph:
            continue
        op = graph[n][0]
        if op == "buf":
            continue
        for inp in graph[n][1:]:
            if inp in real:
                inputs.add(inp)
            else:
                fused.add(inp)
                stack.append(inp)
    fused.discard(R)
    return {
        'out': R,
        'fused': sorted(fused),
        'inputs': sorted(inputs),
    }


def schedule(graph, outputs):
    real = _realized(graph, outputs)
    topo = _topo_order(graph, outputs)
    kernels = []
    for n in topo:
        if n in real and n in graph and graph[n][0] != "buf":
            kernels.append(_build_kernel(graph, n, real))
    return kernels


def stage_count(graph, outputs):
    topo = _topo_order(graph, outputs)
    reachable = set(topo)
    real = _realized(graph, outputs)
    out_set = set(outputs)
    count = 0
    for n in real:
        if n in reachable and n in graph:
            if graph[n][0] != "buf" and n not in out_set:
                count += 1
    return count


def recompute_count(graph, outputs):
    kernels = schedule(graph, outputs)
    total = sum(len(k['fused']) for k in kernels)
    distinct = set()
    for k in kernels:
        distinct.update(k['fused'])
    return total - len(distinct)