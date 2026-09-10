def robot_simulation(commands, obstacles):
    # directions: N, E, S, W
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    obstacle_set = set(map(tuple, obstacles))
    
    x, y = 0, 0
    d = 0  # facing north
    max_dist = 0
    
    for cmd in commands:
        if cmd == -2:      # turn left
            d = (d - 1) % 4
        elif cmd == -1:    # turn right
            d = (d + 1) % 4
        else:              # move forward k steps
            for _ in range(cmd):
                nx, ny = x + dx[d], y + dy[d]
                if (nx, ny) in obstacle_set:
                    break  # stop, skip remaining steps
                x, y = nx, ny
                dist = x*x + y*y
                if dist > max_dist:
                    max_dist = dist
    
    return max_dist