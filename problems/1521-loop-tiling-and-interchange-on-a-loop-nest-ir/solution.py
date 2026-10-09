import itertools


def split(loops, exprs, var, size):
    new_loops = []
    for name, extent in loops:
        if name == var:
            assert extent % size == 0, "size must divide the loop extent"
            new_loops.append((var + "o", extent // size))
            new_loops.append((var + "i", size))
        else:
            new_loops.append((name, extent))

    new_exprs = []
    for expr in exprs:
        e = dict(expr)
        k = e.pop(var, 0)
        if k:
            e[var + "o"] = e.get(var + "o", 0) + k * size
            e[var + "i"] = e.get(var + "i", 0) + k
        if "" not in e:
            e[""] = 0
        new_exprs.append(e)

    return new_loops, new_exprs


def interchange(loops, order):
    by_name = {name: extent for name, extent in loops}
    assert set(order) == set(by_name), "order must be a permutation of the loop names"
    return [(name, by_name[name]) for name in order]


def iteration_space(loops, exprs):
    names = [name for name, _ in loops]
    extents = [extent for _, extent in loops]
    result = []
    for values in itertools.product(*(range(e) for e in extents)):
        env = dict(zip(names, values))
        tup = []
        for expr in exprs:
            total = 0
            for v, k in expr.items():
                if v == "":
                    total += k
                else:
                    total += k * env[v]
            tup.append(total)
        result.append(tuple(tup))
    return sorted(result)


def reuse_distance(loops, expr, var):
    return expr.get(var, 0)