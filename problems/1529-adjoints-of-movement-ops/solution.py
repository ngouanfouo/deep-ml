import numpy as np


def apply(op, arg, x):
    if op == 'reshape':
        return np.reshape(x, arg)
    if op == 'expand':
        return np.broadcast_to(x, arg)
    if op == 'permute':
        return np.transpose(x, arg)
    if op == 'flip':
        return np.flip(x, axis=arg)
    if op == 'pad':
        return np.pad(x, arg)
    if op == 'shrink':
        slices = tuple(slice(start, end) for start, end in arg)
        return x[slices]
    if op == 'sum':
        axes = arg
        if isinstance(axes, int):
            axes = (axes,)
        return np.sum(x, axis=axes, keepdims=True)
    raise ValueError(f"unknown op {op}")


def adjoint(op, arg, x_shape, g):
    if op == 'reshape':
        return np.reshape(g, x_shape)
    if op == 'expand':
        out_shape = g.shape
        ndim_out = len(out_shape)
        ndim_in = len(x_shape)
        if ndim_out > ndim_in:
            g = g.sum(axis=tuple(range(ndim_out - ndim_in)))
        axes = []
        for i in range(ndim_in):
            if x_shape[i] == 1 and g.shape[i] > 1:
                axes.append(i)
        if axes:
            g = g.sum(axis=tuple(axes), keepdims=True)
        return np.reshape(g, x_shape)
    if op == 'permute':
        inv = tuple(np.argsort(arg))
        return np.transpose(g, inv)
    if op == 'flip':
        return np.flip(g, axis=arg)
    if op == 'pad':
        slices = []
        for k in range(len(x_shape)):
            lo = arg[k][0]
            hi = lo + x_shape[k]
            slices.append(slice(lo, hi))
        return g[tuple(slices)]
    if op == 'shrink':
        pads = []
        for k in range(len(x_shape)):
            start, end = arg[k]
            after = x_shape[k] - end
            pads.append((start, after))
        return np.pad(g, pads)
    if op == 'sum':
        return np.broadcast_to(g, x_shape)
    raise ValueError(f"unknown op {op}")


def check_adjoint(op, arg, x_shape, rng):
    x = rng.standard_normal(x_shape)
    out = apply(op, arg, x)
    y = rng.standard_normal(out.shape)
    left = np.sum(y * out)
    right = np.sum(adjoint(op, arg, x_shape, y) * x)
    return float(abs(left - right))