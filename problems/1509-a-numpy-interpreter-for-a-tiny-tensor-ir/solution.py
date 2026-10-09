import numpy as np


def evaluate(node, bufs, cache=None):
    if cache is None:
        cache = {}
    if node in cache:
        return cache[node]

    op = node[0]

    if op == "buf":
        result = np.asarray(bufs[node[1]], dtype=np.float32)
    elif op == "const":
        result = np.float32(node[1])
    elif op == "add":
        result = evaluate(node[1], bufs, cache) + evaluate(node[2], bufs, cache)
    elif op == "mul":
        result = evaluate(node[1], bufs, cache) * evaluate(node[2], bufs, cache)
    elif op == "max":
        result = np.maximum(evaluate(node[1], bufs, cache),
                            evaluate(node[2], bufs, cache))
    elif op == "neg":
        result = -evaluate(node[1], bufs, cache)
    elif op == "exp":
        result = np.exp(evaluate(node[1], bufs, cache))
    elif op == "recip":
        result = np.float32(1.0) / evaluate(node[1], bufs, cache)
    elif op == "reshape":
        result = evaluate(node[1], bufs, cache).reshape(node[2])
    elif op == "expand":
        result = np.broadcast_to(evaluate(node[1], bufs, cache), node[2])
    elif op == "permute":
        result = evaluate(node[1], bufs, cache).transpose(node[2])
    elif op == "flip":
        result = np.flip(evaluate(node[1], bufs, cache), node[2])
    elif op == "pad":
        result = np.pad(evaluate(node[1], bufs, cache), node[2])
    elif op == "shrink":
        a = evaluate(node[1], bufs, cache)
        result = a[tuple(slice(s, e) for s, e in node[2])]
    elif op == "sum":
        result = evaluate(node[1], bufs, cache).sum(
            axis=node[2], keepdims=True, dtype=np.float32)
    elif op == "amax":
        result = evaluate(node[1], bufs, cache).max(axis=node[2], keepdims=True)
    else:
        raise ValueError(f"unknown op: {op}")

    result = np.asarray(result, dtype=np.float32)
    cache[node] = result
    return result


def eval_count(node, bufs):
    cache = {}
    evaluate(node, bufs, cache)
    return len(cache)