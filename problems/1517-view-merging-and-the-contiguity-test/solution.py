import math


def strides_for(shape):
    shape = tuple(shape)
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return tuple(st)


def is_contiguous(shape, strides):
    shape = tuple(shape)
    strides = tuple(strides)
    expected = strides_for(shape)
    for sz, s, e in zip(shape, strides, expected):
        if sz == 1:
            continue
        if s != e:
            return False
    return True


def merge_reshape(shape, strides, new_shape):
    shape = tuple(shape)
    strides = tuple(strides)
    new_shape = tuple(new_shape)

    old_total = 1
    for s in shape:
        old_total *= s
    new_total = 1
    for s in new_shape:
        new_total *= s
    if old_total != new_total:
        return None
    if old_total == 0:
        return None

    result = []
    oi = 0
    ni = 0

    while oi < len(shape) or ni < len(new_shape):
        old_start = oi
        new_start = ni

        # Absorb leading size-1 axes on both sides (they don't affect the product)
        while oi < len(shape) and shape[oi] == 1:
            oi += 1
        while ni < len(new_shape) and new_shape[ni] == 1:
            ni += 1

        if oi < len(shape) and ni < len(new_shape):
            op = shape[oi]; oi += 1
            np = new_shape[ni]; ni += 1
            while op != np:
                if op < np:
                    if oi >= len(shape):
                        return None
                    op *= shape[oi]; oi += 1
                else:
                    if ni >= len(new_shape):
                        return None
                    np *= new_shape[ni]; ni += 1
            # Absorb trailing size-1 axes into this group
            while oi < len(shape) and shape[oi] == 1:
                oi += 1
            while ni < len(new_shape) and new_shape[ni] == 1:
                ni += 1
        else:
            # One side ran out of non-1 axes; any remaining axes must be size 1
            while oi < len(shape):
                if shape[oi] != 1:
                    return None
                oi += 1
            while ni < len(new_shape):
                if new_shape[ni] != 1:
                    return None
                ni += 1

        # Mergeability of old axes in this group
        old_axes = [(shape[i], strides[i])
                    for i in range(old_start, oi) if shape[i] != 1]
        if not old_axes:
            merged_stride = 0
        else:
            for k in range(len(old_axes) - 1):
                if old_axes[k][1] != old_axes[k + 1][1] * old_axes[k + 1][0]:
                    return None
            merged_stride = old_axes[-1][1]

        # Split the merged block into the new axes
        new_sizes = new_shape[new_start:ni]
        k = len(new_sizes)
        out = [0] * k
        suffix = 1
        for j in range(k - 1, -1, -1):
            out[j] = 0 if new_sizes[j] == 1 else merged_stride * suffix
            suffix *= new_sizes[j]
        result.extend(out)

    return tuple(result)


def reshape_cost(shape, strides, new_shape):
    if merge_reshape(shape, strides, new_shape) is not None:
        return 0
    n = 1
    for s in shape:
        n *= s
    return n