def bounds(expr, ranges):
    if isinstance(expr, bool):
        raise ValueError("bool is not a valid IR node")
    if isinstance(expr, int):
        return (expr, expr)
    if isinstance(expr, str):
        lo, hi = ranges[expr]
        return (lo, hi - 1)

    if isinstance(expr, tuple):
        op = expr[0]

        if op == "add":
            a = bounds(expr[1], ranges)
            b = bounds(expr[2], ranges)
            return (a[0] + b[0], a[1] + b[1])

        if op == "mul":
            a = bounds(expr[1], ranges)
            b = bounds(expr[2], ranges)
            corners = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
            return (min(corners), max(corners))

        if op == "idiv":
            a = bounds(expr[1], ranges)
            c = expr[2]
            # positive divisor: floor division is monotonic
            return (a[0] // c, a[1] // c)

        if op == "mod":
            a = bounds(expr[1], ranges)
            c = expr[2]
            if a[0] >= 0 and a[1] < c:
                return (a[0], a[1])
            return (0, c - 1)

        if op == "max":
            a = bounds(expr[1], ranges)
            b = bounds(expr[2], ranges)
            return (max(a[0], b[0]), max(a[1], b[1]))

        if op == "lt":
            a = bounds(expr[1], ranges)
            b = bounds(expr[2], ranges)
            # always true: every left value < every right value
            if a[1] < b[0]:
                return (1, 1)
            # always false: every left value >= every right value
            if a[0] >= b[1]:
                return (0, 0)
            return (0, 1)

        if op == "and":
            a = bounds(expr[1], ranges)
            b = bounds(expr[2], ranges)
            # certain 0 if either operand is certainly 0
            if a[1] == 0 or b[1] == 0:
                return (0, 0)
            # certain 1 if both operands are certainly nonzero
            if a[0] >= 1 and b[0] >= 1:
                return (1, 1)
            return (0, 1)

        if op == "where":
            c = bounds(expr[1], ranges)
            if c == (1, 1):
                return bounds(expr[2], ranges)
            if c == (0, 0):
                return bounds(expr[3], ranges)
            a = bounds(expr[2], ranges)
            b = bounds(expr[3], ranges)
            return (min(a[0], b[0]), max(a[1], b[1]))

    raise ValueError(f"cannot bound {expr!r}")


def fold_compare(expr, ranges):
    if not isinstance(expr, tuple) or expr[0] not in ("lt", "and"):
        return None
    lo, hi = bounds(expr, ranges)
    if lo == hi:
        return bool(lo)
    return None