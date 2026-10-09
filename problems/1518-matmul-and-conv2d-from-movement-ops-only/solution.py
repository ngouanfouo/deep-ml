import numpy as np


def matmul_mov(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]
    batch = a.shape[:-2]

    # a -> (..., M, 1, K)
    a_r = a.reshape(batch + (M, 1, K))
    # b -> (1, ..., 1, N, K)
    b_r = b.T.reshape((1,) * len(batch) + (N, K))

    full = batch + (M, N, K)
    prod = np.broadcast_to(a_r, full) * np.broadcast_to(b_r, full)
    return prod.sum(axis=-1)


def conv2d_mov(x, w, pad=0):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)

    N, Cin, H, W = x.shape
    Cout, Cin_w, kh, kw = w.shape
    assert Cin == Cin_w

    Ho = H + 2 * pad - kh + 1
    Wo = W + 2 * pad - kw + 1

    if pad > 0:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x

    full = (N, Cout, Cin, Ho, Wo)
    acc = np.zeros(full, dtype=np.float32)

    for dh in range(kh):
        for dw in range(kw):
            window = xp[:, :, dh:dh + Ho, dw:dw + Wo]      # (N, Cin, Ho, Wo)
            window_r = window.reshape(N, 1, Cin, Ho, Wo)    # broadcast over Cout
            w_tap = w[:, :, dh, dw].reshape(1, Cout, Cin, 1, 1)  # broadcast over N, Ho, Wo
            acc += np.broadcast_to(window_r, full) * np.broadcast_to(w_tap, full)

    return acc.sum(axis=2)


def count_views(shape_a, shape_b):
    shape_a = tuple(shape_a)
    shape_b = tuple(shape_b)
    M = shape_a[-2]
    K = shape_a[-1]
    N = shape_b[-1]
    shape = shape_a[:-2] + (M, N, K)
    count = 1
    for s in shape:
        count *= s
    return shape, count