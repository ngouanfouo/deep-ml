def best_meeting_point(grid):
    # Collect all home coordinates
    homes = [(r, c) for r, row in enumerate(grid)
                     for c, val in enumerate(row) if val == 1]
    
    if not homes:
        return 0
    
    # Median of row coordinates and column coordinates independently
    rows = sorted(r for r, _ in homes)
    cols = sorted(c for _, c in homes)
    
    r_star = rows[len(rows) // 2]
    c_star = cols[len(cols) // 2]
    
    # Total Manhattan distance from all homes to (r_star, c_star)
    total = sum(abs(r - r_star) + abs(c - c_star) for r, c in homes)
    return total