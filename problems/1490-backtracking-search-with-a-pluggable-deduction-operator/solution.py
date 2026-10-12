import numpy as np


def pin(x, cell, digit):
    r, c = cell
    new = x.copy()
    new[r, c, :] = False
    new[r, c, digit - 1] = True
    return new


def choose_cell(x, rng, strategy):
    n = x.shape[0]
    open_cells = []
    for r in range(n):
        for c in range(n):
            cnt = int(np.sum(x[r, c]))
            if cnt >= 2:
                open_cells.append((cnt, r, c))
    if not open_cells:
        return None

    if strategy == "mrv":
        min_cnt = min(cnt for cnt, _, _ in open_cells)
        for cnt, r, c in open_cells:
            if cnt == min_cnt:
                return (r, c)
    elif strategy == "random":
        cells = [(r, c) for _, r, c in open_cells]
        idx = int(rng.choice(len(cells)))
        return cells[idx]
    else:
        raise ValueError(f"unknown strategy: {strategy}")


def solve(x, deduce, rng, strategy="mrv", max_nodes=100000):
    nodes = 0

    def dfs(state):
        nonlocal nodes
        if nodes >= max_nodes:
            return None
        nodes += 1
        new_state = deduce(state)
        sums = np.sum(new_state, axis=2)
        if np.any(sums == 0):
            return None
        if np.all(sums == 1):
            return new_state

        cell = choose_cell(new_state, rng, strategy)
        if cell is None:
            return None

        r, c = cell
        candidates = np.where(new_state[r, c])[0] + 1  # 1-based digits
        if strategy == "random":
            candidates = candidates[rng.permutation(len(candidates))]

        for d in candidates:
            pinned = pin(new_state, (r, c), int(d))
            res = dfs(pinned)
            if res is not None:
                return res
            if nodes >= max_nodes:
                return None
        return None

    res = dfs(x)
    return res, nodes


def count_solutions(x, deduce, limit):
    rng = np.random.default_rng(0)  # mrv does not use rng
    count = 0

    def dfs(state):
        nonlocal count
        if count >= limit:
            return
        new_state = deduce(state)
        sums = np.sum(new_state, axis=2)
        if np.any(sums == 0):
            return
        if np.all(sums == 1):
            count += 1
            return

        cell = choose_cell(new_state, rng, "mrv")
        if cell is None:
            return

        r, c = cell
        candidates = np.where(new_state[r, c])[0] + 1
        for d in candidates:
            pinned = pin(new_state, (r, c), int(d))
            dfs(pinned)
            if count >= limit:
                return

    dfs(x)
    return count