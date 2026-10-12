def launch_dims(shape, block=(32, 4, 1)):
    L = len(shape)
    ex = shape[-1] if L >= 1 else 1
    ey = shape[-2] if L >= 2 else 1
    ez = shape[-3] if L >= 3 else 1

    bx, by, bz = block
    bx = min(bx, ex)
    by = min(by, ey)
    bz = min(bz, ez)

    gx = -(-ex // bx)
    gy = -(-ey // by)
    gz = -(-ez // bz)

    return (gx, gy, gz), (bx, by, bz)


def threads_launched(grid, block):
    return grid[0] * block[0] * grid[1] * block[1] * grid[2] * block[2]


def utilization(shape, block=(32, 4, 1)):
    grid, blk = launch_dims(shape, block)
    launched = threads_launched(grid, blk)
    useful = 1
    for s in shape:
        useful *= s
    return useful / launched


def global_index(block_id, thread_id, block):
    return tuple(block_id[i] * block[i] + thread_id[i] for i in range(3))


def in_bounds(gidx, shape):
    L = len(shape)
    ex = shape[-1] if L >= 1 else 1
    ey = shape[-2] if L >= 2 else 1
    ez = shape[-3] if L >= 3 else 1
    gx, gy, gz = gidx
    return gx < ex and gy < ey and gz < ez


def best_block(shape, candidates):
    best = None
    best_util = -1.0
    for cand in candidates:
        u = utilization(shape, cand)
        if u > best_util:
            best_util = u
            best = cand
    return best