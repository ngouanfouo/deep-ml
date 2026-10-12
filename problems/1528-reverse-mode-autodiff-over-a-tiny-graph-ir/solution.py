import numpy as np


def _topo(node):
    """Return nodes in topological order (post-order)."""
    visited = set()
    order = []
    stack = [(node, False)]
    while stack:
        n, processed = stack.pop()
        if processed:
            order.append(n)
            continue
        if n in visited:
            continue
        visited.add(n)
        stack.append((n, True))
        op = n[0]
        if op in ('add', 'mul', 'max'):
            stack.append((n[1], False))
            stack.append((n[2], False))
        elif op in ('neg', 'exp', 'recip', 'reshape', 'expand', 'permute', 'sum'):
            stack.append((n[1], False))
    return order


def forward(node, bufs, cache):
    if node in cache:
        return cache[node]
    op = node[0]
    if op == 'buf':
        val = np.asarray(bufs[node[1]], dtype=np.float32)
    elif op == 'const':
        val = np.float32(node[1])
    elif op == 'add':
        val = forward(node[1], bufs, cache) + forward(node[2], bufs, cache)
    elif op == 'mul':
        val = forward(node[1], bufs, cache) * forward(node[2], bufs, cache)
    elif op == 'neg':
        val = -forward(node[1], bufs, cache)
    elif op == 'exp':
        val = np.exp(forward(node[1], bufs, cache))
    elif op == 'recip':
        val = 1.0 / forward(node[1], bufs, cache)
    elif op == 'max':
        a = forward(node[1], bufs, cache)
        b = forward(node[2], bufs, cache)
        val = np.maximum(a, b)
    elif op == 'reshape':
        val = np.reshape(forward(node[1], bufs, cache), node[2])
    elif op == 'expand':
        a = forward(node[1], bufs, cache)
        val = np.array(np.broadcast_to(a, node[2]), dtype=np.float32)
    elif op == 'permute':
        val = np.transpose(forward(node[1], bufs, cache), node[2])
    elif op == 'sum':
        axes = node[2]
        if isinstance(axes, int):
            axes = (axes,)
        val = np.sum(forward(node[1], bufs, cache), axis=axes, keepdims=True)
    else:
        raise ValueError(f"unknown op {op}")
    cache[node] = val
    return val


def backward(loss, bufs, wrt):
    cache = {}
    forward(loss, bufs, cache)          # populate cache
    order = _topo(loss)

    grad = {}
    loss_val = forward(loss, bufs, cache)
    grad[loss] = np.ones_like(loss_val, dtype=np.float32)

    def add_grad(node, contrib):
        target_shape = forward(node, bufs, cache).shape
        # Reduce contribution to target_shape if broadcasting occurred
        if contrib.shape != target_shape:
            # Sum over leading extra dimensions
            ndim_diff = len(contrib.shape) - len(target_shape)
            if ndim_diff > 0:
                contrib = contrib.sum(axis=tuple(range(ndim_diff)))
            # Sum over dimensions where target has size 1 but contrib has >1
            for i, (c, t) in enumerate(zip(contrib.shape, target_shape)):
                if t == 1 and c > 1:
                    contrib = contrib.sum(axis=i, keepdims=True)
        if node not in grad:
            grad[node] = np.zeros(target_shape, dtype=np.float32)
        grad[node] += contrib

    for node in reversed(order):
        if node not in grad:
            continue
        g = grad[node]
        op = node[0]
        if op == 'add':
            add_grad(node[1], g)
            add_grad(node[2], g)
        elif op == 'mul':
            a, b = node[1], node[2]
            va = forward(a, bufs, cache)
            vb = forward(b, bufs, cache)
            add_grad(a, g * vb)
            add_grad(b, g * va)
        elif op == 'neg':
            add_grad(node[1], -g)
        elif op == 'exp':
            out = forward(node, bufs, cache)
            add_grad(node[1], g * out)
        elif op == 'recip':
            out = forward(node, bufs, cache)
            add_grad(node[1], -g * out * out)
        elif op == 'max':
            a, b = node[1], node[2]
            va = forward(a, bufs, cache)
            vb = forward(b, bufs, cache)
            mask_a = (va >= vb).astype(np.float32)
            mask_b = (va < vb).astype(np.float32)
            add_grad(a, g * mask_a)
            add_grad(b, g * mask_b)
        elif op == 'reshape':
            a = node[1]
            va_shape = forward(a, bufs, cache).shape
            add_grad(a, np.reshape(g, va_shape))
        elif op == 'expand':
            a = node[1]
            va = forward(a, bufs, cache)
            in_shape = va.shape
            out_shape = g.shape
            ndim_out = len(out_shape)
            ndim_in = len(in_shape)
            offset = ndim_out - ndim_in
            axes = []
            for i in range(ndim_out):
                if i < offset:
                    axes.append(i)
                else:
                    if in_shape[i - offset] == 1:
                        axes.append(i)
            if axes:
                g_sum = np.sum(g, axis=tuple(axes), keepdims=True)
            else:
                g_sum = g
            g_sum = np.reshape(g_sum, in_shape)
            add_grad(a, g_sum)
        elif op == 'permute':
            a = node[1]
            perm = node[2]
            inv_perm = tuple(np.argsort(perm))
            add_grad(a, np.transpose(g, inv_perm))
        elif op == 'sum':
            a = node[1]
            va = forward(a, bufs, cache)
            add_grad(a, np.broadcast_to(g, va.shape))

    result = {}
    for name in wrt:
        buf_node = ('buf', name)
        if buf_node in grad:
            result[name] = grad[buf_node]
        else:
            result[name] = np.zeros_like(bufs[name], dtype=np.float32)
    return result


def _forward64(node, bufs):
    op = node[0]
    if op == 'buf':
        return np.asarray(bufs[node[1]], dtype=np.float64)
    elif op == 'const':
        return np.float64(node[1])
    elif op == 'add':
        return _forward64(node[1], bufs) + _forward64(node[2], bufs)
    elif op == 'mul':
        return _forward64(node[1], bufs) * _forward64(node[2], bufs)
    elif op == 'neg':
        return -_forward64(node[1], bufs)
    elif op == 'exp':
        return np.exp(_forward64(node[1], bufs))
    elif op == 'recip':
        return 1.0 / _forward64(node[1], bufs)
    elif op == 'max':
        return np.maximum(_forward64(node[1], bufs), _forward64(node[2], bufs))
    elif op == 'reshape':
        return np.reshape(_forward64(node[1], bufs), node[2])
    elif op == 'expand':
        return np.array(np.broadcast_to(_forward64(node[1], bufs), node[2]), dtype=np.float64)
    elif op == 'permute':
        return np.transpose(_forward64(node[1], bufs), node[2])
    elif op == 'sum':
        axes = node[2]
        if isinstance(axes, int):
            axes = (axes,)
        return np.sum(_forward64(node[1], bufs), axis=axes, keepdims=True)
    else:
        raise ValueError(f"unknown op {op}")


def numeric_gradient(loss, bufs, name, eps=1e-3):
    base = np.asarray(bufs[name], dtype=np.float64)
    grad = np.zeros_like(base, dtype=np.float64)
    flat = base.ravel()
    for i in range(flat.size):
        orig = flat[i]

        bufs_plus = dict(bufs)
        buf_plus = base.copy()
        buf_plus.ravel()[i] = orig + eps
        bufs_plus[name] = buf_plus
        loss_plus = _forward64(loss, bufs_plus)
        loss_plus = np.sum(loss_plus)

        bufs_minus = dict(bufs)
        buf_minus = base.copy()
        buf_minus.ravel()[i] = orig - eps
        bufs_minus[name] = buf_minus
        loss_minus = _forward64(loss, bufs_minus)
        loss_minus = np.sum(loss_minus)

        grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad