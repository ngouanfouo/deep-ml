def substitute(expr, mapping):
    if isinstance(expr, str):
        return mapping.get(expr, expr)
    if not isinstance(expr, tuple):
        return expr

    op = expr[0]
    if op in ("add", "mul"):
        return (op, substitute(expr[1], mapping), substitute(expr[2], mapping))
    if op == "load":
        return ("load", expr[1], substitute(expr[2], mapping))
    return expr


def unroll(update, acc, var, u):
    cur = acc
    for j in range(u):
        idx = ("add", ("mul", var + "o", u), j)
        cur = substitute(update, {var: idx, acc: cur})
    return cur


def evaluate(expr, env, bufs):
    if isinstance(expr, str):
        return env[expr]
    if not isinstance(expr, tuple):
        return expr

    op = expr[0]
    if op == "add":
        return evaluate(expr[1], env, bufs) + evaluate(expr[2], env, bufs)
    if op == "mul":
        return evaluate(expr[1], env, bufs) * evaluate(expr[2], env, bufs)
    if op == "load":
        return bufs[expr[1]][evaluate(expr[2], env, bufs)]

    raise ValueError(f"unknown operation: {op}")


def run_loop(update, acc_init, var, K, bufs):
    acc = evaluate(acc_init, {}, bufs)
    for k in range(K):
        acc = evaluate(update, {"acc": acc, var: k}, bufs)
    return acc


def run_unrolled(update, acc_init, var, K, u, bufs):
    acc = evaluate(acc_init, {}, bufs)
    unrolled = unroll(update, "acc", var, u)
    outer_var = var + "o"

    for ko in range(K // u):
        acc = evaluate(unrolled, {"acc": acc, outer_var: ko}, bufs)

    return acc


def count_loads(expr):
    if not isinstance(expr, tuple):
        return 0

    op = expr[0]
    if op in ("add", "mul"):
        return count_loads(expr[1]) + count_loads(expr[2])
    if op == "load":
        return 1 + count_loads(expr[2])
    return 0