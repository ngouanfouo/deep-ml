from collections import deque

def min_fill_time(grid):
    # grid is a list of lists of ints (1 = source, 0 = empty, 2 = wall)
    # return the minimum time to fill all empty cells, or -1 if impossible
    if not grid or not grid[0]:
        return 0
    
    R, C = len(grid), len(grid[0])
    q = deque()
    empty_count = 0
    
    # Initialize: all sources at time 0
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 1:
                q.append((i, j, 0))
            elif grid[i][j] == 0:
                empty_count += 1
    
    if empty_count == 0:
        return 0
    
    max_time = 0
    filled = 0
    
    while q:
        r, c, t = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0:
                grid[nr][nc] = 1          # mark as filled
                filled += 1
                max_time = max(max_time, t + 1)
                q.append((nr, nc, t + 1))
    
    return max_time if filled == empty_count else -1