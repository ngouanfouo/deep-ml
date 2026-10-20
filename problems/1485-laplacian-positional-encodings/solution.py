import numpy as np


def laplacian(A, normalized=True):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    degrees = A.sum(axis=1)

    if normalized:
        inv_sqrt = np.zeros(n, dtype=float)
        nonzero = degrees > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        L = np.eye(n) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])
    else:
        L = np.diag(degrees) - A

    return L


def canonical_sign(U):
    U = np.array(U, dtype=float, copy=True)
    for j in range(U.shape[1]):
        col = U[:, j]
        idx = np.where(np.abs(col) > 1e-8)[0]
        if len(idx) > 0 and col[idx[0]] < 0:
            U[:, j] *= -1.0
    return U


def laplacian_pe(A, k, normalized=True):
    L = laplacian(A, normalized=normalized)
    eigvals, eigvecs = np.linalg.eigh(L)

    keep = np.where(eigvals >= 1e-8)[0]
    selected = keep[:k]

    eigenvalues = eigvals[selected]
    P = canonical_sign(eigvecs[:, selected])

    return eigenvalues, P


def pe_distances(P):
    P = np.asarray(P, dtype=float)
    diff = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def random_sign_flip(P, signs):
    P = np.asarray(P, dtype=float)
    signs = np.asarray(signs, dtype=float)
    return P * signs[None, :]