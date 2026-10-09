import numpy as np


def shape_after(shape0, ops):
    shape = tuple(shape0)
    for op in ops:
        name = op[0]
        if name in ("reshape", "expand"):
            shape = tuple(op[1])
        elif name == "permute":
            perm = op[1]
            shape = tuple(shape[p] for p in perm)
        elif name == "flip":
            pass
        elif name == "pad":
            shape = tuple(s + lo + hi for s, (lo, hi) in zip(shape, op[1]))
        elif name == "shrink":
            shape = tuple(e - s for s, e in op[1])
        else:
            raise ValueError(f"unknown op: {name}")
    return shape


def _strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def _flatten(idx, shape):
    return sum(i * s for i, s in zip(idx, _strides(shape)))


def _unflatten(f, shape):
    st = _strides(shape)
    return tuple((f // st[i]) % shape[i] for i in range(len(shape)))


def source_index(shape0, ops, idx):
    shapes = [tuple(shape0)]
    for op in ops:
        shapes.append(shape_after(shapes[-1], [op]))

    idx = tuple(idx)
    valid = True

    for k in range(len(ops) - 1, -1, -1):
        op = ops[k]
        old_shape = shapes[k]
        new_shape = shapes[k + 1]
        name = op[0]

        if name == "reshape":
            f = _flatten(idx, new_shape)
            idx = _unflatten(f, old_shape)
        elif name == "expand":
            new_idx = list(idx)
            for a in range(len(old_shape)):
                if old_shape[a] == 1 and new_shape[a] != 1:
                    new_idx[a] = 0
            idx = tuple(new_idx)
        elif name == "permute":
            perm = op[1]
            new_idx = [0] * len(old_shape)
            for i, p in enumerate(perm):
                new_idx[p] = idx[i]
            idx = tuple(new_idx)
        elif name == "flip":
            axes = set(op[1])
            new_idx = list(idx)
            for a in axes:
                new_idx[a] = old_shape[a] - 1 - new_idx[a]
            idx = tuple(new_idx)
        elif name == "pad":
            new_idx = []
            for a, (lo, hi) in enumerate(op[1]):
                if not (lo <= idx[a] < lo + old_shape[a]):
                    valid = False
                new_idx.append(idx[a] - lo)
            idx = tuple(new_idx)
        elif name == "shrink":
            new_idx = []
            for a, (start, end) in enumerate(op[1]):
                new_idx.append(idx[a] + start)
            idx = tuple(new_idx)
        else:
            raise ValueError(f"unknown op: {name}")

    return _flatten(idx, shape0), valid


def materialize(shape0, ops, buf):
    out_shape = shape_after(shape0, ops)
    out = np.zeros(out_shape, dtype=np.float32)
    buf = np.asarray(buf, dtype=np.float32)
    for idx in np.ndindex(out_shape):
        f, valid = source_index(shape0, ops, idx)
        if valid:
            out[idx] = buf[f]
    return out