def term_bounds(terms, ranges):
    k = terms.get("", 0)
    vmin = k
    vmax = k
    for v, a in terms.items():
        if v == "":
            continue
        lo, hi = ranges[v]
        hi -= 1  # inclusive upper bound
        if a >= 0:
            vmin += a * lo
            vmax += a * hi
        else:
            vmin += a * hi
            vmax += a * lo
    return vmin, vmax


def fold_div(terms, c, ranges):
    assert c > 0
    k = terms.get("", 0)

    quotient = {"": 0}
    if k % c == 0:
        quotient[""] = k // c
        remainder = {"": 0}
    else:
        remainder = {"": k}

    for v, a in terms.items():
        if v == "":
            continue
        if a % c == 0:
            q = a // c
            if q != 0:
                quotient[v] = q
        else:
            remainder[v] = a

    rmin, rmax = term_bounds(remainder, ranges)
    if 0 <= rmin and rmax < c:
        return quotient, None
    return quotient, remainder


def fold_mod(terms, c, ranges):
    assert c > 0
    remainder = {}
    for v, a in terms.items():
        if v == "":
            continue
        if a % c == 0:
            continue
        remainder[v] = a
    k = terms.get("", 0)
    if k % c == 0:
        remainder[""] = 0
    else:
        remainder[""] = k
    rmin, rmax = term_bounds(remainder, ranges)
    exact = (0 <= rmin and rmax < c)
    return remainder, exact


def unflatten(shape, flat_terms, ranges):
    n = len(shape)
    strides = [1] * n
    for j in range(n - 2, -1, -1):
        strides[j] = strides[j + 1] * shape[j + 1]

    result = []
    for j in range(n):
        s = strides[j]
        q, r = fold_div(flat_terms, s, ranges)
        if r is not None:
            result.append("div/mod")
            continue
        rem, exact = fold_mod(q, shape[j], ranges)
        if exact:
            result.append(rem)
        else:
            result.append("div/mod")
    return result