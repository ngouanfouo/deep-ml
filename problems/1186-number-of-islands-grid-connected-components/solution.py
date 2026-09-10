def count_islands(grid):
    if not grid or not grid[0]:
        return 0
    
    R, C = len(grid), len(grid[0])
    visited = [[False] * C for _ in range(R)]
    count = 0
    
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 1 and not visited[i][j]:
                count += 1
                # Iterative DFS flood fill
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                        if 0 <= nr < R and 0 <= nc < C \
                                and grid[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    
    return count