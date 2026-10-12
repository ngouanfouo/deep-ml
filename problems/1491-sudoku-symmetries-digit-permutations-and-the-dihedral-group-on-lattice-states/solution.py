import numpy as np


def apply_dihedral(x, k):
    if k < 4:
        return np.rot90(x, k, axes=(0, 1))
    else:
        x = np.flip(x, axis=1)
        return np.rot90(x, k - 4, axes=(0, 1))


def inverse_dihedral(k):
    probe = np.arange(8 * 8).reshape(8, 8, 1)
    target = apply_dihedral(probe, k)
    for j in range(8):
        if np.array_equal(apply_dihedral(target, j), probe):
            return j
    return None


def permute_digits(x, perm):
    n = len(perm)
    out = np.empty_like(x)
    for d in range(n):
        out[..., perm[d]] = x[..., d]
    return out


def inverse_perm(perm):
    perm = np.asarray(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def random_symmetry(rng, n):
    k = int(rng.integers(8))
    perm = rng.permutation(n)
    return (k, perm)


def apply_symmetry(x, sym):
    k, perm = sym
    x = permute_digits(x, perm)
    x = apply_dihedral(x, k)
    return x


def invert_symmetry(x, sym):
    k, perm = sym
    x = apply_dihedral(x, inverse_dihedral(k))
    x = permute_digits(x, inverse_perm(perm))
    return x


def is_valid_solution(grid, box_rows, box_cols):
    grid = np.asarray(grid)
    n = grid.shape[0]
    target = set(range(1, n + 1))

    for r in range(n):
        if set(grid[r, :].tolist()) != target:
            return False
    for c in range(n):
        if set(grid[:, c].tolist()) != target:
            return False
    for br in range(0, n, box_rows):
        for bc in range(0, n, box_cols):
            box = grid[br:br + box_rows, bc:bc + box_cols].ravel().tolist()
            if set(box) != target:
                return False
    return True