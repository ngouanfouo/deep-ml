import numpy as np


def emulate(kernel, shape, block, out, bufs):
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

    skipped = 0
    for bz_i in range(gz):
        for by_i in range(gy):
            for bx_i in range(gx):
                for tz in range(bz):
                    for ty in range(by):
                        for tx in range(bx):
                            gidx0 = bx_i * bx + tx
                            gidx1 = by_i * by + ty
                            gidx2 = bz_i * bz + tz
                            if gidx0 < ex and gidx1 < ey and gidx2 < ez:
                                val = kernel((gidx0, gidx1, gidx2), bufs)
                                flat = gidx2 * (ey * ex) + gidx1 * ex + gidx0
                                out.flat[flat] = val
                            else:
                                skipped += 1
    return out, skipped


def warp_transactions(addresses, line_bytes=128, elem_bytes=4):
    return len({(a * elem_bytes) // line_bytes for a in addresses})


def coalescing_cost(index_fn, shape, block):
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

    total = 0
    for bz_i in range(gz):
        for by_i in range(gy):
            for bx_i in range(gx):
                warp_addrs = []
                for tid in range(bx * by * bz):
                    tx = tid % bx
                    ty = (tid // bx) % by
                    tz = tid // (bx * by)
                    gidx0 = bx_i * bx + tx
                    gidx1 = by_i * by + ty
                    gidx2 = bz_i * bz + tz
                    if gidx0 < ex and gidx1 < ey and gidx2 < ez:
                        addr = index_fn((gidx0, gidx1, gidx2))
                        warp_addrs.append(addr)
                    if len(warp_addrs) == 32:
                        total += warp_transactions(warp_addrs)
                        warp_addrs = []
                if warp_addrs:
                    total += warp_transactions(warp_addrs)
    return total