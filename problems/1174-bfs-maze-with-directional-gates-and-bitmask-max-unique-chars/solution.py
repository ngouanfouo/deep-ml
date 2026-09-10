from collections import deque
from functools import lru_cache

def fn(grid, words):
    # ---------- Task 1: Maze BFS ----------
    R = len(grid)
    C = len(grid[0]) if R else 0
    start = end = None
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'E':
                end = (i, j)
    
    maze_result = -1
    if start is not None and end is not None:
        q = deque([(start, 0)])
        visited = {start}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == end:
                maze_result = d
                break
            ch = grid[r][c]
            if ch == '>':
                moves = [(r, c + 1)]
            elif ch == '<':
                moves = [(r, c - 1)]
            else:
                moves = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for nr, nc in moves:
                if 0 <= nr < R and 0 <= nc < C \
                        and grid[nr][nc] != '#' \
                        and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), d + 1))
    
    # ---------- Task 2: Max concat length with no repeated letters ----------
    best = {}  # letter-mask -> longest word using exactly that mask
    for w in words:
        m = 0
        ok = True
        for ch in w:
            b = 1 << (ord(ch) - ord('a'))
            if m & b:
                ok = False
                break
            m |= b
        if ok:
            if m not in best or best[m] < len(w):
                best[m] = len(w)
    items = list(best.items())
    
    @lru_cache(maxsize=None)
    def dp(mask):
        res = 0
        for m, l in items:
            if mask & m == 0:
                cand = l + dp(mask | m)
                if cand > res:
                    res = cand
        return res
    
    words_result = dp(0) if items else 0
    
    return (maze_result, words_result)