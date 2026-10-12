import numpy as np


def abstract(solutions, n=None):
    if not solutions:
        if n is None:
            raise ValueError("n required for empty solutions")
        return np.zeros((n, n, n), dtype=bool)
    n = len(solutions[0])
    one_hot = np.eye(n, dtype=bool)
    result = np.zeros((n, n, n), dtype=bool)
    for grid in solutions:
        g = np.asarray(grid)
        result |= one_hot[g - 1]
    return result


def gamma_count(x):
    counts = np.sum(x, axis=2)
    result = 1
    for c in counts.ravel():
        result *= int(c)
    return result


def is_consistent(solution, x):
    g = np.asarray(solution)
    n = g.shape[0]
    for r in range(n):
        for c in range(n):
            if not x[r, c, g[r, c] - 1]:
                return False
    return True


def alpha_target(x, solutions, prev=None):
    consistent = [s for s in solutions if is_consistent(s, x)]
    if consistent:
        return x & abstract(consistent)
    if prev is not None:
        return x & prev
    return np.zeros_like(x, dtype=bool)


def round_trip_extra(solutions):
    if not solutions:
        return 0
    # distinct solutions
    distinct = set()
    for s in solutions:
        distinct.add(tuple(tuple(row) for row in s))
    return gamma_count(abstract(solutions)) - len(distinct)