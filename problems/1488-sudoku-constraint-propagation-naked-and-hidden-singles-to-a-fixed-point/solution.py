import numpy as np


def sudoku_units(n, box_rows, box_cols):
    units = []
    # n rows
    for r in range(n):
        units.append([(r, c) for c in range(n)])
    # n columns
    for c in range(n):
        units.append([(r, c) for r in range(n)])
    # n boxes in row-major order of boxes
    for br in range(0, n, box_rows):
        for bc in range(0, n, box_cols):
            units.append([(r, c)
                          for r in range(br, br + box_rows)
                          for c in range(bc, bc + box_cols)])
    return units


def peers(n, box_rows, box_cols):
    units = sudoku_units(n, box_rows, box_cols)
    pm = {cell: set() for cell in
          ((r, c) for r in range(n) for c in range(n))}
    for unit in units:
        s = set(unit)
        for cell in unit:
            pm[cell] |= s - {cell}
    return {cell: sorted(v) for cell, v in pm.items()}


def naked_singles(x, peer_map):
    new = x.copy()
    n = x.shape[0]
    for r in range(n):
        for c in range(n):
            candidates = np.where(x[r, c])[0]
            if len(candidates) == 1:
                d = int(candidates[0])
                for (pr, pc) in peer_map[(r, c)]:
                    new[pr, pc, d] = False
    return new


def hidden_singles(x, units):
    new = x.copy()
    n = x.shape[0]
    for unit in units:
        for d in range(n):
            cells_with_d = [cell for cell in unit if x[cell[0], cell[1], d]]
            if len(cells_with_d) == 1:
                r, c = cells_with_d[0]
                # pin this cell to digit d (clear all other candidates)
                new[r, c, :] = False
                new[r, c, d] = True
    return new


def propagate(x, units, peer_map):
    rounds = 0
    while True:
        new = naked_singles(x, peer_map)
        new = hidden_singles(new, units)
        if np.array_equal(new, x):
            break
        rounds += 1
        x = new
        if np.any(np.sum(x, axis=2) == 0):
            break
    return x, rounds