def _is_const(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _local(e):
    op = e[0]

    if op == "add":
        a, b = e[1], e[2]
        # constant folding
        if _is_const(a) and _is_const(b):
            return a + b
        # x + 0 -> x, 0 + x -> x
        if _is_const(b) and b == 0:
            return a
        if _is_const(a) and a == 0:
            return b
        # canonical: c + x -> x + c
        if _is_const(a) and not _is_const(b):
            return ("add", b, a)
        # chained: (x + c1) + c2 -> x + (c1 + c2)
        if _is_const(b) and isinstance(a, tuple) and a[0] == "add" and _is_const(a[2]):
            return ("add", a[1], a[2] + b)
        # right assoc: x + (y + c) -> (x + y) + c
        if isinstance(b, tuple) and b[0] == "add" and _is_const(b[2]) and not _is_const(a):
            return ("add", ("add", a, b[1]), b[2])
        # (x + c) + y -> (x + y) + c
        if isinstance(a, tuple) and a[0] == "add" and _is_const(a[2]) and not _is_const(b):
            return ("add", ("add", a[1], b), a[2])
        return e

    if op == "mul":
        a, b = e[1], e[2]
        # constant folding
        if _is_const(a) and _is_const(b):
            return a * b
        # x * 1 -> x, 1 * x -> x
        if _is_const(b) and b == 1:
            return a
        if _is_const(a) and a == 1:
            return b
        # x * 0 -> 0, 0 * x -> 0
        if _is_const(b) and b == 0:
            return 0
        if _is_const(a) and a == 0:
            return 0
        # canonical: c * x -> x * c
        if _is_const(a) and not _is_const(b):
            return ("mul", b, a)
        # chained: (x * c1) * c2 -> x * (c1 * c2)
        if _is_const(b) and isinstance(a, tuple) and a[0] == "mul" and _is_const(a[2]):
            return ("mul", a[1], a[2] * b)
        # distribute constant over sum: (u + v) * c -> u*c + v*c
        if _is_const(b) and isinstance(a, tuple) and a[0] == "add":
            return ("add", ("mul", a[1], b), ("mul", a[2], b))
        return e

    if op == "neg":
        a = e[1]
        if _is_const(a):
            return -a
        if isinstance(a, tuple) and a[0] == "neg":
            return a[1]
        return e

    return e


def _simplify(expr):
    if not isinstance(expr, tuple):
        return expr

    # bottom-up: children first
    e = (expr[0],) + tuple(_simplify(c) for c in expr[1:])

    # apply local rules to a fixed point; after any change, re-simplify children
    # (a rewrite like distribution introduces new subexpressions that need work).
    seen = set()
    while True:
        e2 = _local(e)
        if e2 == e:
            return e
        if not isinstance(e2, tuple):
            return e2
        e = (e2[0],) + tuple(_simplify(c) for c in e2[1:])
        if e in seen:      # safety net; shouldn't trigger for these rules
            return e
        seen.add(e)


def simplify(expr):
    return _simplify(expr)


def size(expr):
    if isinstance(expr, tuple):
        return 1 + sum(size(c) for c in expr[1:])
    return 1