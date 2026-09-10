import math

def graph_reachable(points, start, target, r):
    # points: list of [x, y]
    # start, target: integer indices
    # r: float distance threshold
    n = len(points)
    if start == target:
        return True
    if not (0 <= start < n) or not (0 <= target < n):
        return False
    
    r_sq = r * r
    visited = [False] * n
    visited[start] = True
    stack = [start]
    
    while stack:
        i = stack.pop()
        xi, yi = points[i]
        for j in range(n):
            if not visited[j]:
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy <= r_sq:
                    if j == target:
                        return True
                    visited[j] = True
                    stack.append(j)
    
    return False